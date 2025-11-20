import torch
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from skimage.metrics import structural_similarity as skimage_ssim
import numpy as np

def _safe_sort_desc(tensor):
    vals, idx = torch.sort(tensor, descending=True)
    return vals, idx

class H_functions:
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
        nonzero_mask = singulars > 1e-10
        temp[:, nonzero_mask] = temp[:, nonzero_mask] / singulars[nonzero_mask]
        return self.V(self.add_zeros(temp))

class GeneralH(H_functions):
    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2: vshape = vshape * v.shape[2]
        if len(v.shape) > 3: vshape = vshape * v.shape[3]
        return torch.matmul(M, v.view(v.shape[0], vshape, 1)).view(v.shape[0], M.shape[0])

    def __init__(self, H, channels):
        self.channels = channels
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
        return self._singulars.repeat(self.channels)

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self._V.shape[0], device=vec.device)
        out[:, :self._U.shape[0]] = vec.clone().reshape(vec.shape[0], -1)
        return out

class Inpainting(H_functions):
    def __init__(self, channels, img_dim, missing_indices, device, kernel_size=3, sigma=1.0):
        self.channels = channels
        self.img_dim = img_dim
        self.device = device
        self.kernel_size = kernel_size
        self.sigma = sigma
        total_pixels = channels * img_dim**2
        if missing_indices.shape[0] > total_pixels:
            raise ValueError(f"Missing indices ({missing_indices.shape[0]}) exceed total pixels ({total_pixels})")
        self.missing_indices = missing_indices.to(device)
        self.kept_indices = torch.tensor([i for i in range(total_pixels) if i not in missing_indices], device=device).long()
        self.kernel_2d = self._create_2d_gaussian(kernel_size, sigma).to(device)
        self.kernel_x = self.kernel_2d.sum(dim=0)
        self.kernel_x = self.kernel_x / self.kernel_x.sum()
        self.kernel_y = self.kernel_2d.sum(dim=1)
        self.kernel_y = self.kernel_y / self.kernel_y.sum()

        Hx = self._build_1d_matrix(self.kernel_x, img_dim).to(device)
        Hy = self._build_1d_matrix(self.kernel_y, img_dim).to(device)
        Ux, Sx, Vx = torch.svd(Hx, some=False)
        Uy, Sy, Vy = torch.svd(Hy, some=False)
        self.Ux = Ux
        self.Vx = Vx
        self.Uy = Uy
        self.Vy = Vy
        self._singulars = torch.ger(Sx, Sy).flatten().to(device)
        self._singulars = self._singulars.repeat(channels)
        self._singulars[self._singulars < 1e-10] = 0

        self.mask = torch.ones(total_pixels, device=device)
        self.mask[self.missing_indices] = 0

    def _create_2d_gaussian(self, size, sigma):
        coords_y = torch.arange(size, dtype=torch.float32, device=self.device) - size // 2
        coords_x = torch.arange(size, dtype=torch.float32, device=self.device) - size // 2
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij')
        g = torch.exp(-(grid_y**2 + grid_x**2) / (2 * sigma**2))
        return g / g.sum()

    def _build_1d_matrix(self, kernel, dim):
        H = torch.zeros(dim, dim, device=self.device)
        pad = kernel.shape[0] // 2
        for i in range(dim):
            for j in range(i - pad, i + pad + 1):
                if 0 <= j < dim:
                    H[i, j] = kernel[j - i + pad]
        return H

    def V(self, vec):
        bsz = vec.shape[0]
        vec_4d = vec.view(bsz, self.channels, self.img_dim, self.img_dim)
        vec_merged = vec_4d.reshape(bsz * self.channels, self.img_dim, self.img_dim)
        temp = torch.matmul(vec_merged, self.Vx.T)
        result = torch.matmul(self.Uy, temp)
        result_4d = result.view(bsz, self.channels, self.img_dim, self.img_dim)
        result = result_4d.view(bsz, -1)
        return result * self.mask.unsqueeze(0)

    def Vt(self, vec):
        bsz = vec.shape[0]
        vec_4d = vec.view(bsz, self.channels, self.img_dim, self.img_dim)
        vec_merged = vec_4d.reshape(bsz * self.channels, self.img_dim, self.img_dim)
        temp = torch.matmul(vec_merged, self.Vy)
        result = torch.matmul(self.Vx.T, temp)
        result_4d = result.view(bsz, self.channels, self.img_dim, self.img_dim)
        result = result_4d.view(bsz, -1)
        return result * self.mask.unsqueeze(0)

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
        return vec.clone().reshape(vec.shape[0], -1)

    def H_pinv(self, vec):
        temp = self.Ut(vec)
        singulars = self.singulars()
        nonzero_mask = singulars > 1e-10
        temp[:, nonzero_mask] = temp[:, nonzero_mask] / singulars[nonzero_mask]
        result = self.V(self.add_zeros(temp))
        return result * self.mask.unsqueeze(0)

class EMDenoising(H_functions):
    def __init__(self, channels, img_dim, device, sigma=1.2, attenuation_factor=0.0, diffusion_factor=0.0, diffusion_alpha=1.0):
        self.channels = channels
        self.img_dim = img_dim
        self.device = device
        self.sigma = float(sigma)
        self.attenuation_factor = float(attenuation_factor)
        self.diffusion_factor = float(diffusion_factor)

        size = max(3, int(8 * self.sigma + 1))
        if size % 2 == 0:
            size += 1
        x = torch.arange(size, dtype=torch.float32, device=self.device) - size // 2
        kernel_1d = torch.exp(-0.5 * (x / self.sigma)**2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        Hx = self._build_1d_matrix(kernel_1d, img_dim).to(self.device)
        Hy = self._build_1d_matrix(kernel_1d, img_dim).to(self.device)

        try:
            Ux, Sx, Vx = torch.svd(Hx, some=False)
            Uy, Sy, Vy = torch.svd(Hy, some=False)
        except Exception:
            Ux, Sx, Vx = torch.linalg.svd(Hx, full_matrices=False)
            Uy, Sy, Vy = torch.linalg.svd(Hy, full_matrices=False)

        self.Ux, self.Sx, self.Vx = Ux, Sx, Vx
        self.Uy, self.Sy, self.Vy = Uy, Sy, Vy

        sing = torch.outer(self.Sx, self.Sy).flatten()
        sing_sorted, _ = _safe_sort_desc(sing)
        N = sing_sorted.shape[0]
        idx = torch.arange(N, dtype=torch.float32, device=self.device)
        linear_decay = idx / max(1.0, (N - 1))
        beta = self.diffusion_factor
        alpha = self.attenuation_factor
        adjusted = sing_sorted * (1.0 - alpha * linear_decay.clamp(0.0, 1.0)) * torch.exp(-beta * linear_decay)
        adjusted = torch.clamp(adjusted, min=0.0)
        adjusted[adjusted < 1e-12] = 0.0
        self._singulars = adjusted.repeat(self.channels)

        self.Hx = Hx
        self.Hy = Hy
        self._vec_len = self.channels * img_dim * img_dim
        self.mask = torch.ones(self.channels * img_dim * img_dim, device=self.device)
        self.diffusion_alpha = diffusion_alpha

    def _build_1d_matrix(self, kernel, dim):
        pad = kernel.shape[0] // 2
        H = torch.zeros(dim, dim, device=self.device)
        for i in range(dim):
            for k in range(-pad, pad + 1):
                j = i + k
                if 0 <= j < dim:
                    H[i, j] = kernel[k + pad]
        return H

    def singulars(self):
        return self._singulars

    def U(self, vec):
        bsz = vec.shape[0]
        vec_4d = vec.view(bsz, self.channels, self.img_dim, self.img_dim)
        merged = vec_4d.reshape(bsz * self.channels, self.img_dim, self.img_dim)
        temp = torch.matmul(merged, self.Ux.T)
        out = torch.matmul(self.Uy, temp)
        out = out.view(bsz, self.channels, self.img_dim, self.img_dim)
        return out.view(bsz, -1)

    def Ut(self, vec):
        bsz = vec.shape[0]
        vec_4d = vec.view(bsz, self.channels, self.img_dim, self.img_dim)
        merged = vec_4d.reshape(bsz * self.channels, self.img_dim, self.img_dim)
        temp = torch.matmul(self.Uy.T, merged)
        out = torch.matmul(temp, self.Ux)
        out = out.view(bsz, self.channels, self.img_dim, self.img_dim)
        return out.view(bsz, -1)

    def V(self, vec):
        bsz = vec.shape[0]
        vec_4d = vec.view(bsz, self.channels, self.img_dim, self.img_dim)
        merged = vec_4d.reshape(bsz * self.channels, self.img_dim, self.img_dim)
        temp = torch.matmul(merged, self.Vx.T)
        out = torch.matmul(self.Vy, temp)
        out = out.view(bsz, self.channels, self.img_dim, self.img_dim)
        return out.view(bsz, -1)

    def Vt(self, vec):
        bsz = vec.shape[0]
        vec_4d = vec.view(bsz, self.channels, self.img_dim, self.img_dim)
        merged = vec_4d.reshape(bsz * self.channels, self.img_dim, self.img_dim)
        temp = torch.matmul(self.Vy.T, merged)
        out = torch.matmul(temp, self.Vx)
        out = out.view(bsz, self.channels, self.img_dim, self.img_dim)
        return out.view(bsz, -1)

    def H(self, vec):
        bsz = vec.shape[0]
        v = self.Vt(vec)
        s = self.singulars()
        s = s.to(v.device)
        if v.shape[1] != s.shape[0]:
            minlen = min(v.shape[1], s.shape[0])
            v = v[:, :minlen]
            s = s[:minlen]
        out = v * s.view(1, -1)
        out = self.U(out)
        return out

    def H_pinv(self, vec, reg: float = 1e-6):
        ut = self.Ut(vec)
        s = self.singulars().to(ut.device)
        denom = s.clone() * self.diffusion_alpha
        denom = denom + reg
        nonzero = denom > 0
        res = torch.zeros_like(ut)
        if nonzero.any():
            used = nonzero.nonzero(as_tuple=False).view(-1)
            res[:, used] = ut[:, used] / denom[used].view(1, -1)
        out = self.V(res)
        out = out * self.mask.unsqueeze(0)
        return out

def make_ctf_kernel_1d(
    img_dim: int,
    kernel_size: int,
    pixel_size: float,
    lam: float = 0.0197,
    defocus: float = -15000.0,
    Cs: float = 2.7e7,
    amp_contrast: float = 0.07,
    phase_shift: float = 0.0,
    bfactor: float = 0.0,
    device=None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")

    freq = torch.fft.fftfreq(img_dim, d=pixel_size, device=device)
    k2 = freq ** 2
    k4 = k2 ** 2

    pi = torch.pi
    lam_t = torch.tensor(lam, device=device, dtype=torch.float32)
    defocus_t = torch.tensor(defocus, device=device, dtype=torch.float32)
    Cs_t = torch.tensor(Cs, device=device, dtype=torch.float32)

    chi = pi * lam_t * defocus_t * k2 - 0.5 * pi * Cs_t * (lam_t ** 3) * k4 + phase_shift

    amp = torch.tensor(amp_contrast, device=device, dtype=torch.float32)
    ctf = -(
        torch.sqrt(1.0 - amp ** 2) * torch.sin(chi)
        + amp * torch.cos(chi)
    )

    if bfactor > 0.0:
        ctf = ctf * torch.exp(-(bfactor * k2) / 4.0)

    psf = torch.fft.ifft(ctf).real
    psf = torch.fft.fftshift(psf)

    if kernel_size % 2 == 0:
        kernel_size += 1

    mid = img_dim // 2
    half = kernel_size // 2
    start = max(0, mid - half)
    end = min(img_dim, mid + half + 1)
    kernel = psf[start:end]

    if kernel.shape[0] < kernel_size:
        pad_left = half - (mid - start)
        pad_right = half - (end - mid - 1)
        kernel = F.pad(kernel, (pad_left, pad_right))

    kernel = kernel / kernel.abs().sum().clamp_min(1e-12)
    return kernel.to(device)

class EMDeblurring(H_functions):
    def mat_by_img(self, M, v):
        batch_size = v.shape[0]
        return torch.matmul(
            M,
            v.reshape(batch_size * self.channels, self.img_dim, self.img_dim)
        ).reshape(batch_size, self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        batch_size = v.shape[0]
        return torch.matmul(
            v.reshape(batch_size * self.channels, self.img_dim, self.img_dim),
            M
        ).reshape(batch_size, self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel: torch.Tensor, channels: int, img_dim: int,
                 device, sigma: float = 1.0, alpha: float = 1e-3):
        self.img_dim = img_dim
        self.channels = channels
        self.device = device
        self.alpha = float(alpha)
        self.sigma = float(sigma)

        kernel = kernel.to(device).clone()
        kernel = kernel / kernel.sum().clamp_min(1e-12)

        H_small = torch.zeros(img_dim, img_dim, device=device)
        k_half = kernel.shape[0] // 2
        for i in range(img_dim):
            for j in range(i - k_half, i + k_half + 1):
                if 0 <= j < img_dim:
                    H_small[i, j] = kernel[j - i + k_half]

        U_small, s_small, V_small = torch.svd(H_small, some=False)
        s_small[s_small < 1e-10] = 0

        self.U_small = U_small
        self.V_small = V_small
        self.singulars_small = s_small

        sing2d = torch.outer(s_small, s_small).reshape(img_dim ** 2)
        self._singulars, self._perm = sing2d.sort(descending=True)

    def V(self, vec):
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
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
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
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
        denom = singulars ** 2 + self.alpha * (self.sigma ** 2)
        mask = (singulars > 1e-10)
        temp = temp * (singulars / denom * mask).unsqueeze(0)
        return self.V(temp)

    def singulars(self):
        return self._singulars.repeat(self.channels)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

class SuperResolutionEM(H_functions):
    def __init__(self, channels, img_dim, ratio, device, gaussian_sigma=0.8):  
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        self.device = device
        self.gaussian_sigma = gaussian_sigma 
        self.gaussian_kernel = self._create_gaussian_kernel(5, gaussian_sigma).to(device) 

        H = torch.tensor([[1 / ratio**2] * ratio**2], device=device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def _create_gaussian_kernel(self, size, sigma):
        k = torch.arange(-size // 2 + 1, size // 2 + 1, device=self.device)
        x, y = torch.meshgrid(k, k, indexing='ij')
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size).repeat(self.channels, 1, 1, 1) 

    def H(self, vec):
        img = vec.view(-1, self.channels, self.img_dim, self.img_dim)
        blurred = F.conv2d(img, self.gaussian_kernel, padding=2, groups=self.channels) 
        blurred = blurred.view(vec.shape[0], -1)
        return super().H(blurred) 

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
        self.channels = channels
        self.img_dim = img_dim
        self.device = device
        self.rank = rank
        self.use_prev_img_info = use_prev_img_info
        self.similarity_weight = similarity_weight
        self.feature_guidance_weight = feature_guidance_weight

        self.kernel_3d = self._create_3d_gaussian(kernel_size, sigma_x, sigma_y, sigma_z).to(device)
        self.kernel_2d = self.kernel_3d.sum(dim=0)
        self.kernel_2d = self.kernel_2d / self.kernel_2d.sum()
        self.kernel_x = self.kernel_2d.sum(dim=0)
        self.kernel_x = self.kernel_x / self.kernel_x.sum()
        self.kernel_y = self.kernel_2d.sum(dim=1)
        self.kernel_y = self.kernel_y / self.kernel_y.sum()
        self.kernel_z = self.kernel_3d.sum(dim=(1, 2))
        self.kernel_z = self.kernel_z / self.kernel_z.sum()

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

        self.feature_extractor = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].to(device).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def process(self, patch):
        return patch

    def process_with_prev_info(self, prev_img_info, current_img_info):
        curr_min = current_img_info.min()
        curr_max = current_img_info.max()
        is_grayscale = self.channels == 1

        if is_grayscale:
            prev_img_norm = prev_img_info
            current_img_norm = current_img_info
        else:
            prev_img_norm = (prev_img_info - prev_img_info.min()) / (prev_img_info.max() - prev_img_info.min() + 1e-8) if prev_img_info is not None else None
            current_img_norm = (current_img_info - curr_min) / (curr_max - curr_min + 1e-8)

        if prev_img_norm is None:
            return current_img_info

        prev_np = prev_img_norm[0].permute(1, 2, 0).cpu().numpy()
        curr_np = current_img_norm[0].permute(1, 2, 0).cpu().numpy()
        if is_grayscale:
            ssim_score = skimage_ssim(prev_np.squeeze(), curr_np.squeeze(), data_range=1.0, win_size=3)
        else:
            ssim_score = skimage_ssim(prev_np, curr_np, multichannel=True, channel_axis=2, data_range=1.0, win_size=3)
        adaptive_weight = self.similarity_weight * ssim_score

        with torch.no_grad():
            feature_input_prev = prev_img_norm if is_grayscale else prev_img_norm.repeat(1, 3, 1, 1)[:, :3]
            feature_input_curr = current_img_norm if is_grayscale else current_img_norm.repeat(1, 3, 1, 1)[:, :3]
            prev_features = self.feature_extractor(feature_input_prev)
            curr_features = self.feature_extractor(feature_input_curr)
            prev_features = F.normalize(prev_features, p=2, dim=1)
            curr_features = F.normalize(curr_features, p=2, dim=1)
            feature_sim = F.cosine_similarity(prev_features, curr_features, dim=1).mean()

        combined_img = (1 - adaptive_weight) * current_img_norm + adaptive_weight * prev_img_norm
        feature_guidance = (1 - feature_sim) * (prev_img_norm - current_img_norm).abs().mean(dim=(1, 2, 3), keepdim=True)
        combined_img = combined_img + self.feature_guidance_weight * feature_guidance * (prev_img_norm - combined_img)

        combined_img_4d = combined_img.view(-1, self.channels, self.img_dim, self.img_dim)
        kernel_2d = self.kernel_2d.view(1, 1, self.kernel_2d.shape[0], self.kernel_2d.shape[1]).to(self.device)
        kernel_2d = kernel_2d.repeat(self.channels, 1, 1, 1)
        combined_img_4d = F.conv2d(combined_img_4d, kernel_2d, padding=self.kernel_2d.shape[0]//2, groups=self.channels)
        combined_img = combined_img_4d.view_as(combined_img)

        prev_img_blurred = F.conv2d(prev_img_norm.view(-1, self.channels, self.img_dim, self.img_dim), 
                                    kernel_2d, padding=self.kernel_2d.shape[0]//2, groups=self.channels)
        prev_img_blurred = prev_img_blurred.view_as(prev_img_norm)
        z_weight = self.kernel_z[self.kernel_z.shape[0]//2] * 0.5
        combined_img = combined_img + z_weight * (prev_img_blurred - combined_img)

        combined_img = torch.clamp(combined_img, 0.0, 1.0)
        if not is_grayscale:
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
        nonzero_mask = (svals > 1e-10)
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
