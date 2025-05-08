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
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def enhance_contrast(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def preprocess_image(image):
    gray = enhance_contrast(image)
    edges = cv2.Canny(gray, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) 
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    adaptive_thresh = cv2.adaptiveThreshold(closed, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(adaptive_thresh)
    valid_regions = [i for i in range(1, n_labels) 
                     if stats[i][cv2.CC_STAT_AREA] > 1 
                     and stats[i][cv2.CC_STAT_AREA] < 100
                     and stats[i][cv2.CC_STAT_WIDTH] < gray.shape[1] * 0.8 
                     and stats[i][cv2.CC_STAT_HEIGHT] < gray.shape[0] * 0.8]
    if len(valid_regions) == 0:
        raise ValueError("No valid regions found in the image. Check the preprocessing steps.")
    membrane_mask = np.isin(labels, valid_regions).astype(np.uint8) * 255
    membrane_mask = cv2.erode(membrane_mask, kernel, iterations=1)
    return gray, membrane_mask

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

        self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)

    def process_batch_with_prev_info(self, prev_batch, current_batch, config, patch_size, overlap):
        """
        Process the current batch using information from the previous batch for isotropic_em, patch-wise.
        Returns processed patches and their positions.
        """
        from tools.EMSVD import IsotropicEM
        h_funcs = IsotropicEM(
            channels=config.data.channels,
            img_dim=config.data.image_size,
            device=self.device,
            kernel_size=3,
            sigma_x=1.0,
            sigma_y=1.0,
            sigma_z=2.0,
            use_prev_img_info=True
        )
        # Crop current and previous batches into patches
        current_patches, positions, original_size = self.crop_to_patches(current_batch, patch_size, overlap)
        prev_patches = None
        if prev_batch is not None:
            prev_patches, _, _ = self.crop_to_patches(prev_batch, patch_size, overlap)
            assert len(prev_patches) == len(current_patches), "Mismatch in number of patches between prev and current batches"

        processed_patches = []
        for i, current_patch in enumerate(current_patches):
            prev_patch = None if prev_patches is None else prev_patches[i]
            processed_patch = h_funcs.process_with_prev_info(prev_patch, current_patch)
            processed_patches.append(processed_patch)
        
        return processed_patches, positions, original_size

    def crop_to_patches(self, image, patch_size, overlap):
        """Crop an image into patches with overlap."""
        b, c, h, w = image.shape
        if h < patch_size or w < patch_size:
            raise ValueError(f"Input image size ({h}x{w}) is smaller than patch size ({patch_size}x{patch_size})")
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

    def stitch_patches(self, patches, positions, original_size, patch_size, overlap):
        """Stitch patches back into the original image size with overlap."""
        h, w = original_size
        if not patches:
            return torch.zeros(1, 3, h, w).to(self.device)
        b = patches[0].shape[0]
        c = patches[0].shape[1]
        stitched = torch.zeros(b, c, h, w).to(self.device)
        count_map = torch.zeros(b, c, h, w).to(self.device)
        for patch, (y, x) in zip(patches, positions):
            if y + patch_size > h:
                y_end = h
            else:
                y_end = y + patch_size
            if x + patch_size > w:
                x_end = w
            else:
                x_end = x + patch_size
            patch = patch[:, :, :y_end - y, :x_end - x]
            stitched[:, :, y:y_end, x:x_end] += patch
            count_map[:, :, y:y_end, x:x_end] += 1
        count_map[count_map == 0] = 1
        stitched = stitched / count_map
        high_freq = stitched - torch.nn.functional.avg_pool2d(stitched, 3, stride=1, padding=1)
        stitched = stitched + 0.4 * high_freq 
        return stitched

    def sample_sequence(self, model, cls_fn=None):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)
        print(f'Dataset has size {len(test_dataset)}')

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
        pbar = tqdm.tqdm(val_loader)
        prev_batch = None
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)
            original_shape = x_orig.shape
            batch_size, channels, h, w = x_orig.shape

            # Crop into patches
            patches, positions, original_size = self.crop_to_patches(x_orig, config.data.image_size, overlap)
            degraded_patches = []
            restored_patches = []

            # Handle isotropic_em with previous image information patch-wise
            if deg == 'isotropic_em':
                processed_patches, positions, original_size = self.process_batch_with_prev_info(
                    prev_batch, x_orig, self.config, config.data.image_size, overlap
                )
                patches = processed_patches  # Replace patches with processed ones
            prev_batch = x_orig.clone()
            block_size = config.data.image_size

            for patch in patches:
                if deg == 'inp_em':
                    from tools.EMSVD import Inpainting
                    loaded_image = patch[0].cpu().numpy().transpose(1, 2, 0)
                    gray, membrane_mask = preprocess_image(loaded_image)
                    membrane_mask = (membrane_mask == 255).astype(np.uint8)
                    missing_pixels = torch.nonzero(torch.from_numpy(membrane_mask), as_tuple=False).long()
                    H = config.data.image_size
                    W = config.data.image_size
                    linear_idx = missing_pixels[:, 0] * W + missing_pixels[:, 1]
                    channels = config.data.channels
                    H_W = H * W
                    missing = torch.cat([linear_idx + c * H_W for c in range(channels)], dim=0)
                    H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
                elif deg == 'deno_em':
                    from tools.EMSVD import EMDenoising
                    H_funcs = EMDenoising(config.data.channels, 
                                        self.config.data.image_size,
                                        self.device)
                elif deg == 'isotropic_em':
                    from tools.EMSVD import IsotropicEM
                    H_funcs = IsotropicEM(
                        channels=config.data.channels,
                        img_dim=config.data.image_size,
                        device=self.device,
                        kernel_size=1,
                        sigma_x=1.0,
                        sigma_y=1.0,
                        sigma_z=2.0,
                        use_prev_img_info=True
                    )
                elif deg == 'deblur_em':
                    from tools.EMSVD import EMDeblurring
                    sigma = 0.05
                    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma)**2]))
                    kernel = torch.Tensor([pdf(-5), pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4), pdf(5)]).to(self.device)
                    H_funcs = EMDeblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
                elif deg[:2] == 'sr':
                    blur_by = int(deg[2:])
                    from tools.EMSVD import SuperResolutionEM
                    H_funcs = SuperResolutionEM(config.data.channels, config.data.image_size, blur_by, self.device)
                else:
                    print("ERROR: degradation type not supported")
                    quit()

                y_0 = H_funcs.H(patch)
                y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
                pinv_y_0 = H_funcs.H_pinv(y_0).view(
                    y_0.shape[0],
                    config.data.channels,
                    block_size,
                    block_size
                )
                degraded_patches.append(pinv_y_0)

                x = torch.randn(patch.shape, device=self.device)
                with torch.no_grad():
                    x, _ = self.sample_image(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)
                restored_patches.append(inverse_data_transform(config, x[-1]).to(self.device))

            x_orig_full = self.stitch_patches([inverse_data_transform(config, p) for p in patches], positions, original_size, config.data.image_size, overlap)
            degraded_full = self.stitch_patches(degraded_patches, positions, original_size, config.data.image_size, overlap)
            restored_full = self.stitch_patches(restored_patches, positions, original_size, config.data.image_size, overlap)
            for i in range(batch_size):
                tvu.save_image(x_orig_full[i], os.path.join(self.args.image_folder, f"orig_{idx_so_far + i}.png"))
                tvu.save_image(degraded_full[i], os.path.join(self.args.image_folder, f"y0_{idx_so_far + i}.png"))
                tvu.save_image(restored_full[i], os.path.join(self.args.image_folder, f"{idx_so_far + i}_-1.png"))

                orig = x_orig_full[i]
                recon = restored_full[i]
                mse = torch.mean((recon.to(self.device) - orig) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                avg_psnr += psnr

                # Convert to numpy and permute for SSIM
                recon_np = recon.permute(1, 2, 0).cpu().numpy()
                orig_np = orig.permute(1, 2, 0).cpu().numpy()

                # Determine win_size based on image dimensions
                min_dim = min(orig_np.shape[0], orig_np.shape[1])
                win_size = min(7, min_dim)  # Start with 7, but reduce if necessary
                if win_size % 2 == 0:  # Ensure win_size is odd
                    win_size = win_size - 1
                if win_size < 3:  # Skip SSIM if win_size is too small
                    print(f"Warning: Image {idx_so_far + i} is too small ({orig_np.shape[0]}x{orig_np.shape[1]}). Skipping SSIM.")
                    ssim_val = 0.0
                else:
                    ssim_val = skimage_ssim(
                        orig_np,
                        recon_np,
                        data_range=1.0,
                        multichannel=True,
                        channel_axis=2,
                        win_size=win_size,
                        gaussian_weights=True,
                        sigma=1.5
                    )
                avg_ssim += ssim_val

                lpips_val = self.lpips_fn((orig.unsqueeze(0).to(self.device) * 2 - 1), (recon.unsqueeze(0).to(self.device) * 2 - 1))
                avg_lpips += lpips_val.item()
            idx_so_far += batch_size
            num_samples_done = idx_so_far - args.subset_start
            pbar.set_description(
                "PSNR: %.2f, SSIM: %.4f, LPIPS: %.4f" % (
                    avg_psnr / num_samples_done,
                    avg_ssim / num_samples_done,
                    avg_lpips / num_samples_done
                )
            )

        num_samples = idx_so_far - args.subset_start
        avg_psnr = avg_psnr / num_samples
        avg_ssim = avg_ssim / num_samples
        avg_lpips = avg_lpips / num_samples

        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Total Average SSIM: %.4f" % avg_ssim)
        print("Total Average LPIPS: %.4f" % avg_lpips)
        print("Number of samples: %d" % num_samples)

    def sample_image(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
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

    def sample(self):
        cls_fn = None
        if self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            model_path = "exp/model/MitEM/model_512.pt"
            model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cuda"))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)
        self.sample_sequence(model, cls_fn)


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None):
    with torch.no_grad():
        #setup vectors used in the algorithm
        singulars = H_funcs.singulars()
        Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
        Sigma[:singulars.shape[0]] = singulars
        U_t_y = H_funcs.Ut(y_0)
        Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

        #initialize x_T as given in the paper
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)
        inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device)
        inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
        inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)     

        # implement p(x_T | x_0, y) as given in the paper
        # if eigenvalue is too small, we just treat it as zero (only for init) 
        init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
        init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)
        init_y = init_y.view(*x.size())
        remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
        remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
        init_y = init_y + remaining_s * x
        init_y = init_y / largest_sigmas
        
        #setup iteration variables
        x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        #iterate over the timesteps
        for i, j in tqdm.tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            if cls_fn == None:
                et = model(xt, t)
            else:
                et = model(xt, t, classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
            
            if et.size(1) == 6:
                et = et[:, :3]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            #variational inference conditioned on y
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

            #missing pixels
            Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

            #less noisy than y (after)
            Vt_xt_mod_next[:, cond_after] = \
                V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite] + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
            
            #noisier than y (before)
            Vt_xt_mod_next[:, cond_before] = \
                (Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:, cond_before] + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite])

            #aggregate all 3 cases and give next prediction
            xt_mod_next = H_funcs.V(Vt_xt_mod_next)
            xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))


    return xs, x0_preds