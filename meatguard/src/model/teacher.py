import torch
import torch.nn as nn
import torch.nn.functional as F


def _init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ViTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_r=4, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_r)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hid, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        n = self.norm1(x)
        x = x + self.attn(n, n, n)[0]
        return x + self.mlp(self.norm2(x))


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, patch=2, heads=4, mlp_r=2, drop=0.0):
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim), nn.SiLU(),
        )
        self.proj_in = nn.Conv2d(dim, dim, 1, bias=False)
        self.transformer = nn.Sequential(
            *[ViTBlock(dim, heads, mlp_r=mlp_r, drop=drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Conv2d(dim, dim, 1, bias=False)
        self.patch = patch

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch
        x = self.local_conv(x)
        xp = self.proj_in(x)
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            xp = F.pad(xp, (0, pad_w, 0, pad_h))
        Hp, Wp = xp.shape[2], xp.shape[3]
        nH, nW = Hp // p, Wp // p
        xp = xp.view(B, C, nH, p, nW, p).permute(0, 2, 4, 3, 5, 1).reshape(B * nH * nW, p * p, C)
        xp = self.norm(self.transformer(xp))
        xp = xp.reshape(B, nH, nW, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)
        xp = xp[:, :, :H, :W]
        return x + self.proj_out(xp)


class MobileViTS(nn.Module):
    """
    Project-1 champion: Nano MobileViT from the 8-model benchmark.
    ~160k params. Used as the teacher for knowledge distillation.
    """
    def __init__(self, num_classes=2):
        super().__init__()

        def dw(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch), nn.SiLU(),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch), nn.SiLU(),
            )

        self.features = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(8), nn.SiLU(),
            ),
            dw(8, 16, stride=1),
            dw(16, 24, stride=2),
            dw(24, 24, stride=1),
            dw(24, 32, stride=2),
            MobileViTBlock(32, depth=1, patch=2, heads=2),
            dw(32, 48, stride=2),
            MobileViTBlock(48, depth=1, patch=2, heads=2),
            dw(48, 64, stride=2),
            MobileViTBlock(64, depth=1, patch=2, heads=2),
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(128, num_classes),
        )
        self.apply(_init)

    def forward(self, x):
        return self.head(self.features(x))

    def penultimate_features(self, x):
        """
        Pre-classifier features: the head's projected representation after global
        pooling, before dropout and the final FC layer (shape ``(B, 128)``).

        This is the knowledge-distillation target for feature alignment — the
        student's Stage-3 output is projected to match it via a trainable adaptor
        (see ``KDTrainer``). The head's tail is ``(Dropout, Linear)``, so the head
        is run up to ``[:-2]`` = (Conv, BN, SiLU, GAP, Flatten).
        """
        return self.head[:-2](self.features(x))
