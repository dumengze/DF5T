import os
import logging
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torchvision.utils as tvu
import cv2
from skimage.metrics import structural_similarity as skimage_ssim
import lpips
import random
from tools import get_dataset, data_transform, inverse_data_transform
from DF5T_guided_diffusion import dist_util, logger
from DF5T_guided_diffusion.script_util import create_model
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diffusion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(f"Unsupported beta schedule: {beta_schedule}")
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def enhance_contrast(image, is_grayscale=False):
    try:
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        if is_grayscale:
            if len(image.shape) == 3 and image.shape[2] == 1:
                image = image.squeeze(2)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except ValueError as e:
        logger.error(f"Contrast enhancement error: {str(e)}")
        raise

def preprocess_image(image, target_size=None, is_grayscale=False):
    try:
        if image.size == 0:
            raise ValueError("Input image is empty")
        if is_grayscale and len(image.shape) == 3 and image.shape[2] == 3:
            image = np.mean(image, axis=2)
        if target_size is not None:
            if not isinstance(target_size, (int, tuple)) or (isinstance(target_size, int) and target_size <= 0):
                raise ValueError(f"Invalid target_size: {target_size}")
            image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        # Enhance contrast
        enhanced = enhance_contrast(image, is_grayscale=is_grayscale)
        
        # Edge detection
        gray = enhanced if is_grayscale else cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 10, 250)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        adaptive_thresh = cv2.adaptiveThreshold(closed, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Generate membrane mask
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(adaptive_thresh)
        valid_regions = [i for i in range(1, n_labels)
                         if stats[i][cv2.CC_STAT_AREA] > 1
                         and stats[i][cv2.CC_STAT_AREA] < 100
                         and stats[i][cv2.CC_STAT_WIDTH] < gray.shape[1] * 0.8
                         and stats[i][cv2.CC_STAT_HEIGHT] < gray.shape[0] * 0.8]
        if len(valid_regions) == 0:
            logger.warning("No valid regions found in preprocessed image, using empty mask")
            membrane_mask = np.zeros_like(gray)
        else:
            membrane_mask = np.isin(labels, valid_regions).astype(np.uint8) * 255
            membrane_mask = cv2.erode(membrane_mask, kernel, iterations=1)
        
        # Edge enhancement
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, kernel, iterations=1)
        if is_grayscale:
            enhanced = cv2.addWeighted(enhanced, 1.0, edges, 0.5, 0)
        else:
            enhanced_output = enhanced.copy()
            for c in range(3):
                enhanced_output[:, :, c] = cv2.addWeighted(enhanced[:, :, c], 1.0, edges, 0.5, 0)
            enhanced = enhanced_output
        
        return enhanced, membrane_mask
    except ValueError as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise

def pad_image(image, min_size=256):
    try:
        if not isinstance(image, torch.Tensor) or len(image.shape) != 4:
            raise ValueError("Input image must be a 4D torch tensor")
        b, c, h, w = image.shape
        if h >= min_size and w >= min_size:
            return image, (h, w), (0, 0)
        
        new_h = max(h, min_size)
        new_w = max(w, min_size)
        padded = torch.zeros(b, c, new_h, new_w, device=image.device)
        pad_h = (new_h - h) // 2
        pad_w = (new_w - w) // 2
        padded[:, :, pad_h:pad_h+h, pad_w:pad_w+w] = image
        logger.info(f"Image padded from {h}x{w} to {new_h}x{new_w}")
        return padded, (h, w), (pad_h, pad_w)
    except ValueError as e:
        logger.error(f"Image padding error: {str(e)}")
        raise

def crop_image(image, original_size, pad_offsets, upscale_ratio=1):
    try:
        if not isinstance(image, torch.Tensor) or len(image.shape) != 4:
            raise ValueError("Input image must be a 4D torch tensor")
        h, w = original_size
        pad_h, pad_w = pad_offsets
        h, w = h * upscale_ratio, w * upscale_ratio
        _, _, new_h, new_w = image.shape
        y_start = pad_h * upscale_ratio
        y_end = y_start + h
        x_start = pad_w * upscale_ratio
        x_end = x_start + w
        if y_end > new_h or x_end > new_w:
            logger.warning(f"Crop size {y_end}x{x_end} exceeds image size {new_h}x{new_w}, adjusting")
            y_end = min(y_end, new_h)
            x_end = min(x_end, new_w)
        cropped = image[:, :, y_start:y_end, x_start:x_end]
        logger.info(f"Image cropped from {new_h}x{new_w} to {cropped.shape[2]}x{cropped.shape[3]}")
        return cropped
    except ValueError as e:
        logger.error(f"Image cropping error: {str(e)}")
        raise

def sharpen_edges(image, alpha=2.0, beta=-1.0, is_grayscale=False):
    try:
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        if is_grayscale:
            if len(image.shape) == 3 and image.shape[2] == 1:
                image = image.squeeze(2)
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            sharpened = cv2.addWeighted(image, alpha, blurred, beta, 0)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            v_blurred = cv2.GaussianBlur(v, (5, 5), 0)
            v_sharpened = cv2.addWeighted(v, alpha, v_blurred, beta, 0)
            hsv_sharpened = cv2.merge((h, s, v_sharpened))
            sharpened = cv2.cvtColor(hsv_sharpened, cv2.COLOR_HSV2RGB)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    except ValueError as e:
        logger.error(f"Edge sharpening error: {str(e)}")
        raise

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        try:
            self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
        except ImportError as e:
            logger.error(f"Failed to load LPIPS model: {str(e)}")
            raise

        # Store previous batch patch information
        self.prev_patches = None
        self.prev_positions = None
        self.prev_padded_size = None
        self.prev_batch = None  # Store previous batch data

    def process_batch_with_prev_info(self, prev_batch, current_batch, config):
        try:
            from tools.EMSVD import IsotropicEM
            if not isinstance(current_batch, torch.Tensor) or len(current_batch.shape) != 4:
                raise ValueError("Current batch must be a 4D torch tensor")
            channels = current_batch.shape[1]
            patch_size = config.data.image_size
            overlap = self.args.overlap if hasattr(self.args, 'overlap') else 16

            # Split current batch into patches
            current_patches, current_positions, current_padded_size = self.crop_to_patches(
                current_batch, patch_size, overlap
            )
            processed_patches = []

            # Check if IsotropicEM has process method
            if not hasattr(IsotropicEM, 'process'):
                logger.error("IsotropicEM class lacks process method")
                raise AttributeError("IsotropicEM class lacks process method")

            if prev_batch is None or self.prev_patches is None:
                # If no previous batch or patch info, process current batch independently
                logger.info("No previous batch data or patch info, processing current batch independently")
                h_funcs = IsotropicEM(
                    channels=channels,
                    img_dim=patch_size,
                    device=self.device,
                    kernel_size=3,
                    sigma_x=1.0,
                    sigma_y=1.0,
                    sigma_z=2.0,
                    use_prev_img_info=False
                )
                for patch in current_patches:
                    processed_patches.append(h_funcs.process(patch))
            else:
                # Split previous batch (ensure same splitting parameters)
                prev_patches, prev_positions, prev_padded_size = self.prev_patches, self.prev_positions, self.prev_padded_size

                # Check if patch counts match
                if len(prev_patches) != len(current_patches):
                    logger.warning(f"Previous batch patch count {len(prev_patches)} does not match current {len(current_patches)}")
                    h_funcs = IsotropicEM(
                        channels=channels,
                        img_dim=patch_size,
                        device=self.device,
                        kernel_size=3,
                        sigma_x=1.0,
                        sigma_y=1.0,
                        sigma_z=2.0,
                        use_prev_img_info=False
                    )
                    for patch in current_patches:
                        processed_patches.append(h_funcs.process(patch))
                else:
                    # Process patches with matching
                    for idx, (current_patch, current_pos) in enumerate(zip(current_patches, current_positions)):
                        # Find closest previous batch patch
                        min_dist = float('inf')
                        best_prev_patch = None
                        for prev_patch, prev_pos in zip(prev_patches, prev_positions):
                            dist = ((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)**0.5
                            if dist < min_dist:
                                min_dist = dist
                                best_prev_patch = prev_patch

                        if best_prev_patch is None or min_dist > patch_size:
                            logger.warning(f"Patch {idx} found no matching previous batch patch, processing independently")
                            h_funcs = IsotropicEM(
                                channels=channels,
                                img_dim=patch_size,
                                device=self.device,
                                kernel_size=3,
                                sigma_x=1.0,
                                sigma_y=1.0,
                                sigma_z=2.0,
                                use_prev_img_info=False
                            )
                            processed_patches.append(h_funcs.process(current_patch))
                        else:
                            logger.debug(f"Patch {idx} matched to previous batch patch, distance {min_dist:.2f}")
                            h_funcs = IsotropicEM(
                                channels=channels,
                                img_dim=patch_size,
                                device=self.device,
                                kernel_size=3,
                                sigma_x=1.0,
                                sigma_y=1.0,
                                sigma_z=2.0,
                                use_prev_img_info=True
                            )
                            processed_patches.append(h_funcs.process_with_prev_info(best_prev_patch, current_patch))

            # Update previous batch patch information
            self.prev_patches = current_patches
            self.prev_positions = current_positions
            self.prev_padded_size = current_padded_size
            self.prev_batch = current_batch.clone()  # Update previous batch data

            return processed_patches, current_positions, current_padded_size
        except (ValueError, AttributeError) as e:
            logger.error(f"Batch processing error: {str(e)}")
            raise

    def crop_to_patches(self, image, patch_size, overlap):
        try:
            if not isinstance(image, torch.Tensor) or len(image.shape) != 4:
                raise ValueError("Input image must be a 4D torch tensor")
            if patch_size <= 0 or overlap < 0:
                raise ValueError(f"Invalid patch_size {patch_size} or overlap {overlap}")
            b, c, h, w = image.shape
            if h < patch_size or w < patch_size:
                logger.info(f"Image size {h}x{w} smaller than patch size {patch_size}x{patch_size}, using whole image")
                return [image], [(0, 0)], (h, w)
            
            stride = patch_size - overlap
            patches = []
            positions = []
            num_patches_h = (h - patch_size) // stride + 1
            num_patches_w = (w - patch_size) // stride + 1
            for y in range(num_patches_h):
                for x in range(num_patches_w):
                    y_start = y * stride
                    y_end = y_start + patch_size
                    x_start = x * stride
                    x_end = x_start + patch_size
                    patch = image[:, :, y_start:y_end, x_start:x_end]
                    patches.append(patch)
                    positions.append((y_start, x_start))
            if (h - patch_size) % stride != 0:
                y_start = h - patch_size
                for x in range(num_patches_w):
                    x_start = x * stride
                    x_end = x_start + patch_size
                    patch = image[:, :, y_start:h, x_start:x_end]
                    patches.append(patch)
                    positions.append((y_start, x_start))
            if (w - patch_size) % stride != 0:
                x_start = w - patch_size
                for y in range(num_patches_h):
                    y_start = y * stride
                    y_end = y_start + patch_size
                    patch = image[:, :, y_start:y_end, x_start:w]
                    patches.append(patch)
                    positions.append((y_start, x_start))
            if (h - patch_size) % stride != 0 and (w - patch_size) % stride != 0:
                y_start = h - patch_size
                x_start = w - patch_size
                patch = image[:, :, y_start:h, x_start:w]
                patches.append(patch)
                positions.append((y_start, x_start))
            return patches, positions, (h, w)
        except ValueError as e:
            logger.error(f"Patch cropping error: {str(e)}")
            raise

    def stitch_patches(self, patches, positions, original_size, patch_size, overlap, upscale_ratio=1):
        try:
            if not patches:
                raise ValueError("No patches provided")
            h, w = original_size
            h, w = h * upscale_ratio, w * upscale_ratio
            patch_size = patch_size * upscale_ratio
            if not patches:
                logger.warning("No patches provided, returning zero tensor")
                return torch.zeros(1, patches[0].shape[1], h, w).to(self.device)
            
            b = patches[0].shape[0]
            c = patches[0].shape[1]
            stitched = torch.zeros(b, c, h, w).to(self.device)
            count_map = torch.zeros(b, c, h, w).to(self.device)
            for patch, (y, x) in zip(patches, positions):
                y, x = y * upscale_ratio, x * upscale_ratio
                if patch.shape[2] != patch_size or patch.shape[3] != patch_size:
                    patch = F.interpolate(patch, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                patch = patch[:, :, :y_end - y, :x_end - x]
                stitched[:, :, y:y_end, x:x_end] += patch
                count_map[:, :, y:y_end, x:x_end] += 1
            count_map[count_map == 0] = 1
            stitched = stitched / count_map
            high_freq = stitched - torch.nn.functional.avg_pool2d(stitched, 3, stride=1, padding=1)
            stitched = stitched + 0.4 * high_freq
            stitched = torch.clamp(stitched, 0.0, 1.0)
            logger.info(f"Stitched image to size {stitched.shape[2]}x{stitched.shape[3]}")
            return stitched
        except ValueError as e:
            logger.error(f"Patch stitching error: {str(e)}")
            raise

    def sample_sequence(self, model, cls_fn=None):
        try:
            args, config = self.args, self.config
            dataset, test_dataset = get_dataset(args, config)
            if args.subset_start >= 0 and args.subset_end > 0:
                if args.subset_end <= args.subset_start:
                    raise ValueError("subset_end must be greater than subset_start")
                test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
            else:
                args.subset_start = 0
                args.subset_end = len(test_dataset)
            logger.info(f'Dataset size: {len(test_dataset)}')

            def seed_worker(worker_id):
                worker_seed = args.seed % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            g = torch.Generator()
            g.manual_seed(args.seed)
            val_loader = data.DataLoader(
                test_dataset,
                batch_size=config.sampling.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                worker_init_fn=seed_worker,
                generator=g,
            )

            deg = args.deg
            sigma_0 = args.sigma_0
            overlap = args.overlap if hasattr(args, 'overlap') else 16
            avg_psnr = 0.0
            avg_ssim = 0.0
            avg_lpips = 0.0
            idx_init = args.subset_start
            idx_so_far = args.subset_start
            pbar = tqdm.tqdm(val_loader, disable=True)
            for x_orig, classes in pbar:
                try:
                    x_orig = x_orig.to(self.device)
                    x_orig = data_transform(self.config, x_orig)
                    original_shape = x_orig.shape
                    batch_size, channels, h, w = x_orig.shape

                    # Detect if input is grayscale content
                    input_is_gray = channels == 1
                    if channels == 3:
                        channel_diff = torch.max(torch.abs(x_orig[:, 0] - x_orig[:, 1]) + torch.abs(x_orig[:, 1] - x_orig[:, 2]))
                        if channel_diff < 1e-5:
                            input_is_gray = True
                            logger.info("Detected 3-channel grayscale image, treating as grayscale")
                            # Convert to 1 channel for consistency if needed, but keep channels=3 for model
                    is_grayscale = input_is_gray  # Use this for processing decisions

                    x_orig_padded, original_size, pad_offsets = pad_image(x_orig, min_size=256)
                    block_size = max(x_orig_padded.shape[2], x_orig_padded.shape[3])
                    use_patches = h >= config.data.image_size and w >= config.data.image_size

                    patches, positions, padded_size = self.crop_to_patches(x_orig_padded, config.data.image_size, overlap)
                    degraded_patches = []
                    restored_patches = []
                    upscale_ratio = 1

                    if deg == 'isotropic_em':
                        patches, positions, padded_size = self.process_batch_with_prev_info(
                            self.prev_batch, x_orig_padded, self.config
                        )
                    self.prev_batch = x_orig_padded.clone()  # Update previous batch data

                    for patch_idx, patch in enumerate(patches):
                        try:
                            logger.debug(f"Processing patch {patch_idx + 1}/{len(patches)}, size {patch.shape[2]}x{patch.shape[3]}")

                            if deg == 'inp_em':
                                from tools.EMSVD import Inpainting
                                loaded_image = patch[0].cpu().numpy().transpose(1, 2, 0)
                                if input_is_gray and loaded_image.shape[2] == 3:
                                    loaded_image = np.mean(loaded_image, axis=2)
                                    is_grayscale = True
                                try:
                                    processed, membrane_mask = preprocess_image(loaded_image, is_grayscale=is_grayscale)
                                except ValueError as ve:
                                    logger.warning(f"Patch {patch_idx} preprocessing failed: {str(ve)}, using zero mask")
                                    processed = loaded_image
                                    membrane_mask = np.zeros((patch.shape[2], patch.shape[3]), dtype=np.uint8)
                                membrane_mask = (membrane_mask == 255).astype(np.uint8)
                                missing_pixels = torch.nonzero(torch.from_numpy(membrane_mask), as_tuple=False).long()
                                H = patch.shape[2]
                                W = patch.shape[3]
                                linear_idx = missing_pixels[:, 0] * W + missing_pixels[:, 1]
                                H_W = H * W
                                missing = torch.cat([linear_idx + c * H_W for c in range(channels)], dim=0)
                                H_funcs = Inpainting(channels, H, missing, self.device)
                            elif deg == 'deno_em':
                                from tools.EMSVD import EMDenoising
                                H_funcs = EMDenoising(channels, patch.shape[2], self.device)
                            elif deg == 'isotropic_em':
                                from tools.EMSVD import IsotropicEM
                                H_funcs = IsotropicEM(
                                    channels=channels,
                                    img_dim=patch.shape[2],
                                    device=self.device,
                                    kernel_size=3,
                                    sigma_x=1.0,
                                    sigma_y=1.0,
                                    sigma_z=2.0,
                                    use_prev_img_info=True
                                )
                            elif deg == 'deblur_em':
                                from tools.EMSVD import EMDeblurring
                                sigma = 0.05
                                pdf = lambda x: torch.exp(torch.tensor([-0.5 * (x / sigma)**2]))
                                kernel = torch.tensor([pdf(-5), pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4), pdf(5)]).to(self.device)
                                H_funcs = EMDeblurring(kernel / kernel.sum(), channels, patch.shape[2], self.device)
                            elif deg[:2] == 'sr':
                                blur_by = int(deg[2:])
                                from tools.EMSVD import SuperResolutionEM
                                H_funcs = SuperResolutionEM(channels, patch.shape[2], blur_by, self.device, sigma_0=sigma_0, gaussian_sigma=0.8)
                                upscale_ratio = blur_by
                            else:
                                logger.error(f"Unsupported degradation type: {deg}")
                                raise ValueError(f"Unsupported degradation type: {deg}")

                            y_0 = H_funcs.H(patch)
                            y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
                            pinv_y_0 = H_funcs.H_pinv(y_0).view(
                                y_0.shape[0],
                                channels,
                                patch.shape[2],
                                patch.shape[3]
                            )
                            degraded_patches.append(pinv_y_0)

                            x = torch.randn(patch.shape, device=self.device)
                            with torch.no_grad():
                                x, _ = self.sample_image(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)
                            restored_patch = inverse_data_transform(config, x[-1]).to(self.device)
                            if deg[:2] == 'sr':
                                target_size = (patch.shape[2] * upscale_ratio, patch.shape[3] * upscale_ratio)
                                restored_patch = F.interpolate(
                                    restored_patch,
                                    size=target_size,
                                    mode='bilinear',
                                    align_corners=False
                                )
                            restored_patches.append(restored_patch)

                        except Exception as e:
                            logger.error(f"Error processing patch {patch_idx}: {str(e)}")
                            raise

                    x_orig_full = self.stitch_patches(
                        [inverse_data_transform(config, p) for p in patches],
                        positions,
                        padded_size,
                        config.data.image_size if use_patches else padded_size[0],
                        overlap,
                        upscale_ratio=1
                    )
                    degraded_full = self.stitch_patches(
                        degraded_patches,
                        positions,
                        padded_size,
                        config.data.image_size if use_patches else padded_size[0],
                        overlap,
                        upscale_ratio=1
                    )
                    restored_full = self.stitch_patches(
                        restored_patches,
                        positions,
                        padded_size,
                        config.data.image_size if use_patches else padded_size[0],
                        overlap,
                        upscale_ratio=upscale_ratio
                    )

                    if padded_size != original_size:
                        x_orig_full = crop_image(x_orig_full, original_size, pad_offsets, upscale_ratio=1)
                        degraded_full = crop_image(degraded_full, original_size, pad_offsets, upscale_ratio=1)
                        restored_full = crop_image(restored_full, original_size, pad_offsets, upscale_ratio=upscale_ratio)

                    for i in range(batch_size):
                        try:
                            recon = restored_full[i]
                            recon_np = recon.permute(1, 2, 0).cpu().numpy()
                            if input_is_gray and recon_np.shape[2] == 3:
                                recon_np = np.mean(recon_np, axis=2)
                                is_grayscale = True
                            recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
                            recon_sharpened = sharpen_edges(recon_np, alpha=2.0, beta=-1.0, is_grayscale=is_grayscale)
                            if len(recon_sharpened.shape) == 2:
                                recon = torch.from_numpy(recon_sharpened / 255.0).unsqueeze(0).float().to(self.device)
                            else:
                                recon = torch.from_numpy(recon_sharpened / 255.0).permute(2, 0, 1).float().to(self.device)
                            if input_is_gray and recon.shape[0] == 3:
                                recon = recon.mean(dim=0, keepdim=True)

                            tvu.save_image(x_orig_full[i], os.path.join(self.args.image_folder, f"orig_{idx_so_far + i}.png"))
                            tvu.save_image(degraded_full[i], os.path.join(self.args.image_folder, f"y0_{idx_so_far + i}.png"))
                            tvu.save_image(recon, os.path.join(self.args.image_folder, f"{idx_so_far + i}_-1.png"))

                            orig = x_orig_full[i]
                            if input_is_gray and orig.shape[0] == 3:
                                orig = orig.mean(dim=0, keepdim=True)
                            if deg[:2] == 'sr':
                                orig = F.interpolate(orig.unsqueeze(0), size=(recon.shape[1], recon.shape[2]), mode='bilinear', align_corners=False).squeeze(0)
                            mse = torch.mean((recon.to(self.device) - orig) ** 2)
                            psnr = 10 * torch.log10(1 / mse)
                            avg_psnr += psnr

                            recon_np = recon.permute(1, 2, 0).cpu().numpy()
                            orig_np = orig.permute(1, 2, 0).cpu().numpy()
                            min_dim = min(orig_np.shape[0], orig_np.shape[1])
                            win_size = min(7, min_dim)
                            if win_size % 2 == 0:
                                win_size = win_size - 1
                            if win_size < 3:
                                logger.warning(f"Image {idx_so_far + i} too small ({orig_np.shape[0]}x{orig_np.shape[1]}), skipping SSIM")
                                ssim_val = 0.0
                            else:
                                if input_is_gray:
                                    ssim_val = skimage_ssim(
                                        orig_np.squeeze(), recon_np.squeeze(),
                                        data_range=1.0,
                                        gaussian_weights=True,
                                        sigma=1.5
                                    )
                                else:
                                    ssim_val = skimage_ssim(
                                        orig_np, recon_np,
                                        data_range=1.0,
                                        multichannel=True,
                                        channel_axis=2,
                                        win_size=win_size,
                                        gaussian_weights=True,
                                        sigma=1.5
                                    )
                            avg_ssim += ssim_val

                            orig_input = orig.unsqueeze(0).to(self.device)
                            recon_input = recon.unsqueeze(0).to(self.device)
                            if input_is_gray:
                                if orig_input.shape[1] == 1:
                                    orig_input = orig_input.repeat(1, 3, 1, 1)
                                if recon_input.shape[1] == 1:
                                    recon_input = recon_input.repeat(1, 3, 1, 1)
                            lpips_val = self.lpips_fn(orig_input * 2 - 1, recon_input * 2 - 1)
                            avg_lpips += lpips_val.item()
                        except Exception as e:
                            logger.error(f"Error computing metrics for image {idx_so_far + i}: {str(e)}")
                            continue

                    idx_so_far += batch_size
                    num_samples_done = idx_so_far - args.subset_start
                    pbar.set_description(
                        f"PSNR: {avg_psnr / num_samples_done:.2f}, SSIM: {avg_ssim / num_samples_done:.4f}, LPIPS: {avg_lpips / num_samples_done:.4f}"
                    )

                except Exception as e:
                    logger.error(f"Error processing batch index {idx_so_far}: {str(e)}")
                    continue

            num_samples = idx_so_far - args.subset_start
            if num_samples > 0:
                avg_psnr = avg_psnr / num_samples
                avg_ssim = avg_ssim / num_samples
                avg_lpips = avg_lpips / num_samples
                logger.info(f"Overall average PSNR: {avg_psnr:.2f}")
                logger.info(f"Overall average SSIM: {avg_ssim:.4f}")
                logger.info(f"Overall average LPIPS: {avg_lpips:.4f}")
                logger.info(f"Number of samples: {num_samples}")
            else:
                logger.warning("No samples processed successfully")

        except Exception as e:
            logger.error(f"Sample sequence error: {str(e)}")
            raise

    def sample_image(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        try:
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)
            x = efficient_generalized_steps(
                x, 
                seq, 
                model,
                self.betas, 
                H_funcs, 
                y_0, 
                sigma_0, 
                etaB=self.args.etaB, 
                etaA=self.args.eta, 
                etaC=self.args.eta, 
                cls_fn=cls_fn, 
                classes=classes
            )
            if last:
                x = x[0][-1]
            return x
        except Exception as e:
            logger.error(f"Error generating sample image: {str(e)}")
            raise

    def sample(self):
        try:
            cls_fn = None
            if self.config.model.type == 'openai':
                config_dict = vars(self.config.model)
                model = create_model(**config_dict)
                if self.config.model.use_fp16:
                    model.convert_to_fp16()
                model_path = "exp/model/MitEM/model_256.pt"
                model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cuda"))
                model.to(self.device)
                model.eval()
                model = torch.nn.DataParallel(model)
            self.sample_sequence(model, cls_fn)
        except Exception as e:
            logger.error(f"Sampling error: {str(e)}")
            raise

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None):
    try:
        with torch.no_grad():
            singulars = H_funcs.singulars()
            Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
            Sigma[:singulars.shape[0]] = singulars
            U_t_y = H_funcs.Ut(y_0)
            Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

            largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
            largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
            large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)
            inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device)
            inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
            inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)     

            init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
            init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)
            init_y = init_y.view(*x.size())
            remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
            remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
            init_y = init_y + remaining_s * x
            init_y = init_y / largest_sigmas
            
            x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            x0_preds = []
            xs = [x]

            for i, j in tqdm.tqdm(zip(reversed(seq), reversed(seq_next)), disable=True):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                if cls_fn is None:
                    et = model(xt, t)
                else:
                    et = model(xt, t, classes)
                    et = et[:, :x.shape[1]]
                    et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
                
                if et.size(1) > x.shape[1]:
                    et = et[:, :x.shape[1]]
                
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
                sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
                xt_mod = xt / at.sqrt()[0, 0, 0, 0]
                V_t_x = H_funcs.Vt(xt_mod)
                SVt_x = (V_t_x * Sigma)[:, :U_t_y.shape[1]]
                V_t_x0 = H_funcs.Vt(x0_t)
                SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

                falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)
                cond_before_lite = singulars * sigma_next > sigma_0
                cond_after_lite = singulars * sigma_next < sigma_0
                cond_before = torch.hstack((cond_before_lite, falses))
                cond_after = torch.hstack((cond_after_lite, falses))

                std_nextC = sigma_next * etaC
                sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

                std_nextA = sigma_next * etaA
                sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)
                
                diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

                Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

                Vt_xt_mod_next[:, cond_after] = \
                    V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite] + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
                
                Vt_xt_mod_next[:, cond_before] = \
                    (Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:, cond_before] + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite])

                xt_mod_next = H_funcs.V(Vt_xt_mod_next)
                xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))

        return xs, x0_preds
    except Exception as e:
        logger.error(f"Efficient generalized steps error: {str(e)}")
        raise