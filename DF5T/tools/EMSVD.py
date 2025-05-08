import torch
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from skimage.metrics import structural_similarity as skimage_ssim
import numpy as np

class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """
    def V(self, vec):
        raise NotImplementedError()

    def Vt(self, vec):
        raise NotImplementedError()

    def U(self, vec):
        raise NotImplementedError()

    def Ut(self, vec):
        raise NotImplementedError()

    def singulars(self):
        raise NotImplementedError()

    def add_zeros(self, vec):
        raise NotImplementedError()
    
    def H(self, vec):
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])
    
    def Ht(self, vec):
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))
    
    def H_pinv(self, vec):
        temp = self.Ut(vec)
        singulars = self.singulars()
        nonzero_mask = singulars != 0
        temp[:, nonzero_mask] = temp[:, nonzero_mask] / singulars[nonzero_mask]
        return self.V(self.add_zeros(temp))

class GeneralH(H_functions):
    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2: vshape = vshape * v.shape[2]
        if len(v.shape) > 3: vshape = vshape * v.shape[3]
        return torch.matmul(M, v.view(v.shape[0], vshape, 1)).view(v.shape[0], M.shape[0])

    def __init__(self, H):
        self._U, self._singulars, self._V = torch.svd(H, some=False)
        self._Vt = self._V.transpose(0, 1)
        self._Ut = self._U.transpose(0, 1)
        ZERO = 1e-3
        self._singulars[self._singulars < ZERO] = 0

    def V(self, vec):
        return self.mat_by_vec(self._V, vec.clone())

    def Vt(self, vec):
        return self.mat_by_vec(self._Vt, vec.clone())

    def U(self, vec):
        return self.mat_by_vec(self._U, vec.clone())

    def Ut(self, vec):
        return self.mat_by_vec(self._Ut, vec.clone())

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self._V.shape[0], device=vec.device)
        out[:, :self._U.shape[0]] = vec.clone().reshape(vec.shape[0], -1)
        return out

class Inpainting(H_functions):
    def __init__(self, channels, img_dim, missing_indices, device):
        self.channels = channels
        self.img_dim = img_dim
        self._singulars = torch.ones(channels * img_dim**2 - missing_indices.shape[0]).to(device)
        self.missing_indices = missing_indices.to(device)
        self.kept_indices = torch.Tensor([i for i in range(channels * img_dim**2) if i not in missing_indices]).to(device).long()

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros((vec.shape[0], self.channels * self.img_dim**2), device=vec.device)
        out[:, self.kept_indices] = temp[:, :self.kept_indices.shape[0]]
        out[:, self.missing_indices] = temp[:, self.kept_indices.shape[0]:]
        return out.reshape(vec.shape[0], -1)

    def Vt(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, :self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = temp[:, self.missing_indices]
        return out

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim**2), device=vec.device)
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp

class EMDenoising(H_functions):
    def __init__(self, channels, img_dim, device, alpha=0.7, beta=0.2, sigma=0.05):
        self.channels = channels
        self.img_dim = img_dim
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        total_pixels = channels * img_dim**2
        self._singulars = torch.linspace(1.0, 0.5, total_pixels, device=device)

    def _diffusion(self, x):
        return x * (1 + self.alpha)

    def _frequency_attenuation(self, x):
        x_freq = torch.fft.fft2(x)
        attenuation = torch.linspace(0, 1, self.img_dim, device=self.device)
        attenuation = attenuation.view(1, 1, self.img_dim, 1).expand(-1, -1, self.img_dim, self.img_dim)
        x_freq_attenuated = x_freq * (1 - self.beta * attenuation)
        return torch.fft.ifft2(x_freq_attenuated).real

    def _add_noise(self, x):
        poisson_noise = torch.poisson(x)
        gaussian_noise = torch.randn_like(x) * self.sigma
        return poisson_noise + gaussian_noise

    def V(self, vec):
        return vec.reshape(vec.shape[0], -1)

    def Vt(self, vec):
        return vec.reshape(vec.shape[0], -1)

    def U(self, vec):
        diffused = self._diffusion(vec)
        return diffused.reshape(vec.shape[0], -1)

    def Ut(self, vec):
        restored = vec / (1 + self.alpha)
        return restored.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars * (1 - self.beta)

    def add_zeros(self, vec):
        return vec.reshape(vec.shape[0], -1)

class EMDeblurring(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim, 
                        self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim, 
                        self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, sigma=1.0, alpha=1e-3):
        self.img_dim = img_dim
        self.channels = channels
        self.device = device
        self.alpha = alpha
        self.sigma = sigma
        H_small = torch.zeros(img_dim, img_dim, device=device)
        kernel = kernel / kernel.sum()
        for i in range(img_dim):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2 + 1):
                if j < 0 or j >= img_dim: continue
                H_small[i, j] = kernel[j - i + kernel.shape[0]//2]
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        self.singulars_small[self.singulars_small < 1e-10] = 0
        self._singulars = torch.outer(self.singulars_small, self.singulars_small).reshape(img_dim**2)
        self._singulars, self._perm = self._singulars.sort(descending=True)

    def V(self, vec):
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def H_pinv(self, vec):
        temp = self.Ut(vec)
        singulars = self.singulars()
        denom = singulars**2 + self.alpha * (self.sigma**2)
        mask = (singulars > 1e-10)
        temp = temp * (singulars / denom * mask).unsqueeze(0)
        return self.V(temp)

    def singulars(self):
        return self._singulars.repeat(self.channels)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

class SuperResolutionEM(H_functions):
    def __init__(self, channels, img_dim, ratio, device):
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        H = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(vec.shape[0], self.channels, self.y_dim**2, self.ratio**2, device=vec.device)
        patches[:, :, :, 0] = temp[:, :self.channels * self.y_dim**2].view(vec.shape[0], self.channels, -1)
        for idx in range(self.ratio**2 - 1):
            patches[:, :, :, idx + 1] = temp[:, (self.channels * self.y_dim**2 + idx)::self.ratio**2 - 1].view(vec.shape[0], self.channels, -1)
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        s_full = self.singulars().repeat_interleave(self.ratio**2)
        vec = vec * s_full
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec):
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim ** 2, device=vec.device)
        recon[:, :self.channels * self.y_dim**2] = patches[:, :, :, 0].view(vec.shape[0], self.channels * self.y_dim**2)
        for idx in range(self.ratio**2 - 1):
            recon[:, (self.channels * self.y_dim**2 + idx)::self.ratio**2 - 1] = patches[:, :, :, idx + 1].view(vec.shape[0], self.channels * self.y_dim**2)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio ** 2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp

class IsotropicEM(H_functions):
    def __init__(self, channels, img_dim, device, kernel_size=3, sigma_x=1.0, sigma_y=1.0, sigma_z=2.0, rank=32, use_prev_img_info=True, similarity_weight=0.1, feature_guidance_weight=0.1):
        super().__init__()
        self.channels = channels
        self.img_dim = img_dim
        self.device = device
        self.rank = rank
        self.use_prev_img_info = use_prev_img_info
        self.similarity_weight = similarity_weight
        self.feature_guidance_weight = feature_guidance_weight

        # 三维高斯核，Z 轴 sigma 更大以模拟分辨率较低
        self.kernel_3d = self._create_3d_gaussian(kernel_size, sigma_x, sigma_y, sigma_z).to(device)
        self.kernel_2d = self.kernel_3d.sum(dim=0)  # 沿 Z 轴投影
        self.kernel_2d = self.kernel_2d / self.kernel_2d.sum()
        self.kernel_x = self.kernel_2d.sum(dim=0)  # 投影到 X 轴
        self.kernel_x = self.kernel_x / self.kernel_x.sum()
        self.kernel_y = self.kernel_2d.sum(dim=1)  # 投影到 Y 轴
        self.kernel_y = self.kernel_y / self.kernel_y.sum()
        self.kernel_z = self.kernel_3d.sum(dim=(1, 2))  # Z 轴一维核
        self.kernel_z = self.kernel_z / self.kernel_z.sum()

        # 构建二维卷积矩阵
        Hx = self._build_1D_matrix(self.kernel_x, img_dim).to(device)
        Hy = self._build_1D_matrix(self.kernel_y, img_dim).to(device)
        Ux, Sx, Vx = torch.svd(Hx, some=False)
        Uy, Sy, Vy = torch.svd(Hy, some=False)
        self.Ux = Ux
        self.Vx = Vx
        self.Uy = Uy
        self.Vy = Vy
        self.S = torch.ger(Sx, Sy).flatten()
        self._singulars = self.S.repeat(channels)

        # VGG16 特征提取器
        self.feature_extractor = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].to(device).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def process_with_prev_info(self, prev_img_info, current_img_info):
        """
        Combine previous and current images to simulate Z-axis isotropic restoration.
        """
        # 存储原始范围
        curr_min = current_img_info.min()
        curr_max = current_img_info.max()

        # 归一化到 [0, 1]
        prev_img_norm = (prev_img_info - prev_img_info.min()) / (prev_img_info.max() - prev_img_info.min() + 1e-8) if prev_img_info is not None else None
        current_img_norm = (current_img_info - curr_min) / (curr_max - curr_min + 1e-8)

        # 如果是第一张图像，直接返回
        if prev_img_norm is None:
            return current_img_info

        # 计算 SSIM
        prev_np = prev_img_norm[0].permute(1, 2, 0).cpu().numpy()
        curr_np = current_img_norm[0].permute(1, 2, 0).cpu().numpy()
        ssim_score = skimage_ssim(prev_np, curr_np, multichannel=True, data_range=1.0, win_size=3)
        adaptive_weight = self.similarity_weight * ssim_score

        # VGG16 特征提取
        with torch.no_grad():
            prev_features = self.feature_extractor(prev_img_norm)
            curr_features = self.feature_extractor(current_img_norm)
            prev_features = F.normalize(prev_features, p=2, dim=1)
            curr_features = F.normalize(curr_features, p=2, dim=1)
            feature_sim = F.cosine_similarity(prev_features, curr_features, dim=1).mean()

        # 模拟 Z 轴超分辨率：结合前后帧
        combined_img = (1 - adaptive_weight) * current_img_norm + adaptive_weight * prev_img_norm
        feature_guidance = (1 - feature_sim) * (prev_img_norm - current_img_norm).abs().mean(dim=(1, 2, 3), keepdim=True)
        combined_img = combined_img + self.feature_guidance_weight * feature_guidance * (prev_img_norm - combined_img)

        # 应用二维卷积（三维核投影）
        combined_img_4d = combined_img.view(-1, self.channels, self.img_dim, self.img_dim)
        kernel_2d = self.kernel_2d.view(1, 1, self.kernel_2d.shape[0], self.kernel_2d.shape[1]).to(self.device)
        kernel_2d = kernel_2d.repeat(self.channels, 1, 1, 1)  # 适配多通道
        combined_img_4d = F.conv2d(combined_img_4d, kernel_2d, padding=self.kernel_2d.shape[0]//2, groups=self.channels)
        combined_img = combined_img_4d.view_as(combined_img)

        # 模拟 Z 轴退化修复：应用 Z 轴核的加权
        prev_img_blurred = F.conv2d(prev_img_norm.view(-1, self.channels, self.img_dim, self.img_dim), 
                                    kernel_2d, padding=self.kernel_2d.shape[0]//2, groups=self.channels)
        prev_img_blurred = prev_img_blurred.view_as(prev_img_norm)
        z_weight = self.kernel_z[self.kernel_z.shape[0]//2] * 0.5  # 减小 Z 轴权重
        combined_img = combined_img + z_weight * (prev_img_blurred - combined_img)

        # 裁剪并恢复原始尺度
        combined_img = torch.clamp(combined_img, 0.0, 1.0)
        combined_img = combined_img * (curr_max - curr_min) + curr_min
        
        return combined_img

    def V(self, vec):
        bsz = vec.shape[0]
        vec_4d = vec.view(bsz, self.channels, self.img_dim, self.img_dim)
        vec_merged = vec_4d.reshape(bsz * self.channels, self.img_dim, self.img_dim)
        temp = torch.matmul(vec_merged, self.Vx.T)
        result = torch.matmul(self.Uy, temp)
        result_4d = result.view(bsz, self.channels, self.img_dim, self.img_dim)
        return result_4d.view(bsz, -1)

    def Vt(self, vec):
        bsz = vec.shape[0]
        vec_4d = vec.view(bsz, self.channels, self.img_dim, self.img_dim)
        vec_merged = vec_4d.reshape(bsz * self.channels, self.img_dim, self.img_dim)
        temp = torch.matmul(vec_merged, self.Vy)
        result = torch.matmul(self.Vx.T, temp)
        result_4d = result.view(bsz, self.channels, self.img_dim, self.img_dim)
        return result_4d.view(bsz, -1)

    def U(self, vec):
        bsz = vec.shape[0]
        vec_4d = vec.view(bsz, self.channels, self.img_dim, self.img_dim)
        vec_merged = vec_4d.reshape(bsz * self.channels, self.img_dim, self.img_dim)
        temp = torch.matmul(vec_merged, self.Ux.T)
        result = torch.matmul(self.Uy, temp)
        result_4d = result.view(bsz, self.channels, self.img_dim, self.img_dim)
        return result_4d.view(bsz, -1)

    def Ut(self, vec):
        bsz = vec.shape[0]
        vec_4d = vec.view(bsz, self.channels, self.img_dim, self.img_dim)
        vec_merged = vec_4d.reshape(bsz * self.channels, self.img_dim, self.img_dim)
        temp = torch.matmul(vec_merged, self.Ux)
        result = torch.matmul(self.Uy.T, temp)
        result_4d = result.view(bsz, self.channels, self.img_dim, self.img_dim)
        return result_4d.view(bsz, -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        return vec

    def H(self, vec):
        temp = self.Vt(vec)
        svals = self.singulars()
        temp = temp * svals.unsqueeze(0)[:, :temp.shape[1]]
        return self.U(temp)

    def Ht(self, vec):
        temp = self.Ut(vec)
        svals = self.singulars()
        temp = temp * svals.unsqueeze(0)[:, :temp.shape[1]]
        return self.V(temp)

    def H_pinv(self, vec):
        temp = self.Ut(vec)
        svals = self.singulars()
        nonzero_mask = (svals != 0)
        temp[:, nonzero_mask] = temp[:, nonzero_mask] / svals[nonzero_mask]
        return self.V(temp)

    def _create_3d_gaussian(self, size, sigma_x, sigma_y, sigma_z):
        coords_z = torch.arange(size, dtype=torch.float32, device=self.device) - size // 2
        coords_y = torch.arange(size, dtype=torch.float32, device=self.device) - size // 2
        coords_x = torch.arange(size, dtype=torch.float32, device=self.device) - size // 2
        grid_z, grid_y, grid_x = torch.meshgrid(coords_z, coords_y, coords_x, indexing='ij')
        g = torch.exp(-(grid_z**2 / (2 * sigma_z**2) + grid_y**2 / (2 * sigma_y**2) + grid_x**2 / (2 * sigma_x**2)))
        return g / g.sum()

    def _build_1D_matrix(self, kernel, dim):
        H = torch.zeros(dim, dim, device=self.device)
        pad = kernel.shape[0] // 2
        for i in range(dim):
            for j in range(i - pad, i + pad + 1):
                if 0 <= j < dim:
                    H[i, j] = kernel[j - i + pad]
        return H