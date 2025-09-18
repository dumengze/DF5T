import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import torch as th
from einops import rearrange
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .fp16_util import convert_module_to_f16, convert_module_to_f32


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class TimeEmbedding(nn.Module):
    def __init__(self, time_embed_dim, model_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(int(model_channels), int(time_embed_dim)),
            nn.SiLU(),
            nn.Linear(int(time_embed_dim), int(time_embed_dim)),
        )
    
    def forward(self, timesteps):

        timesteps = timesteps.type(self.time_embed[0].weight.dtype)
        return self.time_embed(timesteps)
    
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)




##########################################################################
## 多类别卷积混合专家（MoME）前馈网络
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):

        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, time_embed_dim, patch_size=8):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.5)
        
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.time_proj = nn.Linear(time_embed_dim, dim)
        
        self.patch_size = patch_size

    def forward(self, x, t_emb):
        b, c, h, w = x.shape
        
        # Project time embeddings and expand to match spatial dimensions
        time_emb = self.time_proj(t_emb)  # Shape: [b, c]
        time_emb = time_emb[:, :, None, None]  # Shape: [b, c, 1, 1]
        time_emb = time_emb.expand(-1, -1, h, w)  # Shape: [b, c, h, w]
        
        # Add time embeddings to the input features
        x = x + time_emb
        
        # 分块操作
        patch_features = F.adaptive_avg_pool2d(x, (self.patch_size, self.patch_size))
        patch_features = F.interpolate(patch_features, size=(h, w), mode='bilinear', align_corners=False)
        x_fused = x + patch_features
        
        # 在融合后的特征上进行q, k, v的投影
        qkv = self.qkv_dwconv(self.qkv(x_fused))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.float() 
        attn = attn.softmax(dim=-1)
        attn = attn.to(x.dtype)
        
        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        out = self.project_out(out)
        
        return out

##########################################################################

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        time_embed_dim,
    ):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)  # Use custom LayerNorm
        self.attn = Attention(dim, num_heads, bias, time_embed_dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)  # Use custom LayerNorm
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)  # 已替换为MoME版本

    def forward(self, x, t_emb):
        dtype = x.dtype
        x = x.float()
        x = x + self.attn(self.norm1(x), t_emb)
        x = x.to(dtype)
        x = x.float() + self.ffn(self.norm2(x))
        x = x.to(dtype)
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class HybridDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HybridDownsample, self).__init__()
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
    
    def forward(self, x):
        x = self.transpose_conv(x)
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)

##########################################################################
## Encoder and Decoder Levels
class EncoderLevel(nn.Module):
    def __init__(self, num_blocks, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, time_embed_dim):
        super(EncoderLevel, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, time_embed_dim)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x, t_emb)
        return x

class DecoderLevel(nn.Module):
    def __init__(self, num_blocks, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, time_embed_dim):
        super(DecoderLevel, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, time_embed_dim)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x, t_emb)
        return x

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        time_embed_dim=384
    ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        # Replace nn.Sequential with EncoderLevel
        self.encoder_level1 = EncoderLevel(num_blocks[0], dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type, time_embed_dim)
        self.down1_2 =  HybridDownsample(in_channels=dim, out_channels=dim*2)
        self.encoder_level2 = EncoderLevel(num_blocks[1], int(dim*2**1), heads[1], ffn_expansion_factor, bias, LayerNorm_type, time_embed_dim)
        self.down2_3 =  HybridDownsample(in_channels=int(dim*2), out_channels=int(dim*4))
        self.encoder_level3 = EncoderLevel(num_blocks[2], int(dim*2**2), heads[2], ffn_expansion_factor, bias, LayerNorm_type, time_embed_dim)
        self.down3_4 =  HybridDownsample(in_channels=int(dim*4), out_channels=int(dim*8))
        self.latent = EncoderLevel(num_blocks[3], int(dim*2**3), heads[3], ffn_expansion_factor, bias, LayerNorm_type, time_embed_dim)
        
        # Decoder levels
        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = EncoderLevel(num_blocks[2], int(dim*2**2), heads[2], ffn_expansion_factor, bias, LayerNorm_type, time_embed_dim)
        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = EncoderLevel(num_blocks[1], int(dim*2**1), heads[1], ffn_expansion_factor, bias, LayerNorm_type, time_embed_dim)
        self.up2_1 = Upsample(int(dim*2**1))
        self.decoder_level1 = EncoderLevel(num_blocks[0], int(dim*2**1), heads[0], ffn_expansion_factor, bias, LayerNorm_type, time_embed_dim)
        self.refinement = EncoderLevel(num_refinement_blocks, int(dim*2**1), heads[0], ffn_expansion_factor, bias, LayerNorm_type, time_embed_dim)
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
        
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.time_embedding = TimeEmbedding(time_embed_dim, dim)
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



    def forward(self, inp_img, timesteps):
        t_emb = self.time_embedding(timesteps.to(self.time_embedding.time_embed[0].weight.dtype))
        
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1, t_emb)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2, t_emb)
        
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3, t_emb)
        
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4, t_emb)
        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3, t_emb)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2, t_emb)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, t_emb)
        
        out_dec_level1 = self.refinement(out_dec_level1, t_emb)

        out_dec_level1 = self.output(out_dec_level1)

        
        return out_dec_level1

    def convert_to_fp16(self):
        self.patch_embed.apply(convert_module_to_f16)
        self.encoder_level1.apply(convert_module_to_f16)
        self.down1_2.apply(convert_module_to_f16)
        self.encoder_level2.apply(convert_module_to_f16)
        self.down2_3.apply(convert_module_to_f16)
        self.encoder_level3.apply(convert_module_to_f16)
        self.down3_4.apply(convert_module_to_f16)
        self.latent.apply(convert_module_to_f16)
        self.up4_3.apply(convert_module_to_f16)
        self.reduce_chan_level3.apply(convert_module_to_f16)
        self.decoder_level3.apply(convert_module_to_f16)
        self.up3_2.apply(convert_module_to_f16)
        self.reduce_chan_level2.apply(convert_module_to_f16)
        self.decoder_level2.apply(convert_module_to_f16)
        self.up2_1.apply(convert_module_to_f16)
        self.decoder_level1.apply(convert_module_to_f16)
        self.refinement.apply(convert_module_to_f16)
        self.output.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.patch_embed.apply(convert_module_to_f32)
        self.encoder_level1.apply(convert_module_to_f32)
        self.down1_2.apply(convert_module_to_f32)
        self.encoder_level2.apply(convert_module_to_f32)
        self.down2_3.apply(convert_module_to_f32)
        self.encoder_level3.apply(convert_module_to_f32)
        self.down3_4.apply(convert_module_to_f32)
        self.latent.apply(convert_module_to_f32)
        self.up4_3.apply(convert_module_to_f32)
        self.reduce_chan_level3.apply(convert_module_to_f32)
        self.decoder_level3.apply(convert_module_to_f32)
        self.up3_2.apply(convert_module_to_f32)
        self.reduce_chan_level2.apply(convert_module_to_f32)
        self.decoder_level2.apply(convert_module_to_f32)
        self.up2_1.apply(convert_module_to_f32)
        self.decoder_level1.apply(convert_module_to_f32)
        self.refinement.apply(convert_module_to_f32)
        self.output.apply(convert_module_to_f32)



class RestormerModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        use_fp16=False,
        num_heads=1,
        num_classes=None,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order
        self.use_fp16 = use_fp16

        time_embed_dim = model_channels * 4
        time_embed_dim = int(time_embed_dim) if time_embed_dim is not None else int(model_channels * 4)
        
        # 如果有类别标签，定义标签Embedding
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        dim = model_channels
        num_blocks = [num_res_blocks] * 4
        heads = [num_heads * (2**i) for i in range(4)]
        self.restormer = Restormer(
            inp_channels=in_channels,
            time_embed_dim=time_embed_dim,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_res_blocks,
            heads=heads,
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',
            dual_pixel_task=False,
        )
        
        if use_fp16:
            self.convert_to_fp16()
        else:
            self.convert_to_fp32()

    def forward(self, x, timesteps, y=None):
        x = x.to(self.dtype)
        t_emb = timestep_embedding(timesteps, self.model_channels).to(self.dtype)
        
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            t_emb = t_emb + self.label_emb(y).to(self.dtype)
        
        if self.use_fp16:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = self.restormer(x, t_emb)
        else:
            out = self.restormer(x, t_emb)
        
        out = out.to(x.dtype)
        
        return out
    
    def convert_to_fp16(self):
        self.restormer.convert_to_fp16()
        if self.num_classes is not None:
            self.label_emb.apply(convert_module_to_f16)
    
    def convert_to_fp32(self):
        self.restormer.convert_to_fp32()
        if self.num_classes is not None:
            self.label_emb.apply(convert_module_to_f32)


import torch
import torch.nn as nn
from einops import rearrange

# Assuming necessary imports and helper functions are defined

class EncoderRestormerModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,  # corresponds to dim in Restormer
        out_channels,
        num_res_blocks,  # corresponds to num_blocks in Restormer
        attention_resolutions,
        use_fp16=False,
        num_head_channels=64,
        pool="adaptive",
        channel_mult=(1, 2, 3, 4),
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.use_fp16 = use_fp16
        self.num_head_channels = num_head_channels
        self.pool = pool
        self.channel_mult = channel_mult

        # Patch embedding
        self.patch_embed = OverlapPatchEmbed(
            in_c=in_channels, embed_dim=model_channels
        )

        # Determine the number of levels based on channel_mult
        self.num_levels = len(channel_mult)

        # Encoder levels
        self.encoder_levels = nn.ModuleList()
        for i in range(self.num_levels):
            dim = model_channels * channel_mult[i]
            num_heads = dim // num_head_channels
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(
                    TransformerBlock(
                        dim=dim,
                        num_heads=num_heads,
                        ffn_expansion_factor=2.66,
                        bias=False,
                        LayerNorm_type='WithBias',
                        time_embed_dim=None,  # Encoder中若无需时间，可以不传
                    )
                )
            self.encoder_levels.append(nn.Sequential(*blocks))

            if i < self.num_levels - 1:
                self.add_module(f"down_{i+1}_{i+2}", HybridDownsample(dim))

        # Pooling layer
        if pool == "adaptive":
            self.pool_layer = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == "spatial":
            self.pool_layer = nn.Sequential(
                nn.Linear(dim * image_size // (2 ** (self.num_levels - 1)) ** 2, 2048),
                nn.ReLU(),
                nn.Linear(2048, out_channels),
            )
        else:
            raise NotImplementedError(f"Pooling method {pool} not implemented")

        # Output layer
        if pool == "adaptive":
            self.out = nn.Linear(dim, out_channels)
        elif pool == "spatial":
            self.out = nn.Identity()  # 已在 pool_layer 中处理

        # Use fp16 if specified
        if use_fp16:
            self.convert_to_fp16()

    def convert_to_fp16(self):
        self.patch_embed.to(torch.float16)
        for module in self.encoder_levels:
            module.to(torch.float16)
        if self.pool == "adaptive":
            self.pool_layer.to(torch.float16)
            self.out.to(torch.float16)

    def forward(self, x):
        # Patch embedding
        h = self.patch_embed(x)

        # Encoder levels
        for i in range(self.num_levels):
            h = self.encoder_levels[i](h)
            if i < self.num_levels - 1:
                h = getattr(self, f"down_{i+1}_{i+2}")(h)

        # Pooling
        if self.pool == "adaptive":
            h = self.pool_layer(h)
            h = h.view(h.size(0), -1)
            output = self.out(h)
        elif self.pool == "spatial":
            # Flatten and pass through pool_layer
            h = h.view(h.size(0), -1)
            output = self.pool_layer(h)
        else:
            raise NotImplementedError(f"Pooling method {self.pool} not implemented")

        return output