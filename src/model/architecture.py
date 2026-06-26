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

class MeatGuardS(nn.Module):
    """
    MeatGuard-S: Hybrid CNN-Transformer Architecture.
    ~3.4M parameters.
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

        # Stage 2: Inverted Residual blocks (Texture hierarchy)
        self.stage2 = nn.Sequential(
            InvertedResidualBlock(64, 64, stride=1),
            InvertedResidualBlock(64, 128, stride=2),
            InvertedResidualBlock(128, 128, stride=1)
        ) # Output: 28x28x128

        # Stage 3: Compact MHSA (Global relations)
        self.stage3 = CompactMHSA(dim=128, heads=4) 
        # Note: Blueprint suggests d=256. If d=256 is needed, add a projection here.
        # We'll keep it at 128 to match Stage 2 for simplicity, or add a 1x1 conv.
        self.s3_proj = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        self.s3_attn = CompactMHSA(dim=256, heads=4) 
        # Output: 28x28x256. Blueprint says 14x14x256.
        # We need a downsample here.
        self.s3_down = nn.MaxPool2d(2) # 28 -> 14

        # Stage 4: MV2 Block + SE Attention (Channel refinement)
        self.stage4 = nn.Sequential(
            InvertedResidualBlock(256, 384, stride=1),
            SEBlock(384)
        ) # Output: 14x14x384. Blueprint says 7x7x384.
        self.s4_down = nn.MaxPool2d(2) # 14 -> 7

        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(384, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        
        # Stage 3
        x = self.s3_proj(x)
        x = self.s3_attn(x)
        x = self.s3_down(x)
        
        # Stage 4
        x = self.stage4(x)
        x = self.s4_down(x)
        
        # Head
        x = self.gap(x)
        return self.classifier(x)

    def get_last_conv_layer_output(self, x):
        """Helper for Grad-CAM: Returns output of Stage 4 before GAP."""
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.s3_proj(x)
        x = self.s3_attn(x)
        x = self.s3_down(x)
        x = self.stage4(x)
        x = self.s4_down(x)
        return x
