import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention."""
    def __init__(self, ch, r=4):
        super().__init__()
        red = max(1, ch // r)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, red),
            nn.SiLU(),
            nn.Linear(red, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x).view(x.size(0), -1, 1, 1)

class InvertedResidualBlock(nn.Module):
    """MobileNetV2 style Inverted Residual Block."""
    def __init__(self, in_ch, out_ch, stride, expand_ratio=4):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        self.use_res_connection = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_ch, hidden_ch, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_ch))
            layers.append(nn.SiLU())

        layers.append(nn.Conv2d(hidden_ch, hidden_ch, 3, stride=stride, padding=1, groups=hidden_ch, bias=False))
        layers.append(nn.BatchNorm2d(hidden_ch))
        layers.append(nn.SiLU())

        layers.append(nn.Conv2d(hidden_ch, out_ch, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connection:
            return x + self.block(x)
        return self.block(x)

class CompactMHSA(nn.Module):
    """Compact Multi-Head Self-Attention for global spatial relations."""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2) 
        
        # QKV projection
        qkv = self.qkv(x_flat).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, -1, self.heads, C // self.heads).transpose(1, 2), qkv)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.proj(out)
        
        # Back to (B, C, H, W)
        return out.transpose(1, 2).reshape(B, C, H, W)

class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension of a (B, C, H, W) map."""
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.LayerNorm(ch)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class TransformerBlock(nn.Module):
    """Pre-norm transformer block on a feature map: x + MHSA(LN(x)), then x + FFN(LN(x)).

    The FFN uses 1x1 convolutions so the block stays on (B, C, H, W) tensors end to end.
    """
    def __init__(self, dim, heads=4, mlp_ratio=2):
        super().__init__()
        hidden = dim * mlp_ratio
        self.norm1 = ChannelLayerNorm(dim)
        self.attn = CompactMHSA(dim, heads)
        self.norm2 = ChannelLayerNorm(dim)
        self.ffn = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, dim, 1))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))

class MeatGuardS(nn.Module):
    """
    MeatGuard-S: Hybrid CNN-Transformer Architecture.
    ~1.14M parameters, ~0.38 GMACs at 224x224 (measured; the original ~3.4M was a planning estimate).
    """
    def __init__(self, num_classes=2):
        super().__init__()

        # Stage 1: Depthwise Separable Conv Blocks (Local features)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        ) # Output: 56x56x64

        # Stage 2: Inverted Residual blocks (Texture hierarchy). Downsampling in the first block and
        # expansion 2 keep the stage at 28x28 and cut its cost ~2x (profiled 2026-10-09).
        self.stage2 = nn.Sequential(
            InvertedResidualBlock(64, 128, stride=2, expand_ratio=2),
            InvertedResidualBlock(128, 128, stride=1, expand_ratio=2),
            InvertedResidualBlock(128, 128, stride=1, expand_ratio=2)
        ) # Output: 28x28x128

        # Stage 3: project to d=256, downsample, then a transformer block (Global relations).
        # Attention runs at 14x14 = 196 tokens; at 28x28 it would cost ~16x more.
        self.s3_proj = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        self.s3_down = nn.MaxPool2d(2) # 28 -> 14
        self.s3_attn = TransformerBlock(dim=256, heads=4) # Output: 14x14x256

        # Stage 4: MV2 Block + SE Attention (Channel refinement)
        self.stage4 = nn.Sequential(
            InvertedResidualBlock(256, 384, stride=1, expand_ratio=2),
            SEBlock(384)
        ) # Output: 14x14x384
        self.s4_down = nn.MaxPool2d(2) # 14 -> 7

        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(384, num_classes)
        )

    def stage3_output(self, x):
        """Stage-3 transformer output (14x14x256); the KD feature-alignment target."""
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.s3_proj(x)
        x = self.s3_down(x)
        return self.s3_attn(x)

    def get_last_conv_layer_output(self, x):
        """Helper for Grad-CAM: Returns output of Stage 4 before GAP (7x7x384)."""
        return self.s4_down(self.stage4(self.stage3_output(x)))

    def forward(self, x):
        x = self.get_last_conv_layer_output(x)
        return self.classifier(self.gap(x))
