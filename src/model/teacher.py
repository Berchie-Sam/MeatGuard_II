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
<<<<<<< Updated upstream
=======


# ---------------------------------------------------------------------------
# ImageNet-pretrained teachers (torchvision) for knowledge distillation.
# The Project-1 MobileViTS above (0.16 M params) is smaller than MeatGuard-S and was trained on a
# random image-level split, so it is kept only for reference; teachers are retrained on the session split.
# ---------------------------------------------------------------------------
from torchvision import models as tvm

TEACHER_ARCHS = {
    # name: (constructor, weights enum)
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "convnext_tiny": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "swin_t": (tvm.swin_t, tvm.Swin_T_Weights.IMAGENET1K_V1),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
}

# Lightweight comparison models, fine-tuned with the teacher recipe (train them with
# `python -m src.train_teacher train --arch <name> --out runs/baselines`, so they are never selected as teacher).
BASELINE_ARCHS = {
    "mobilenet_v3_small": (tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "shufflenet_v2_x1_0": (tvm.shufflenet_v2_x1_0, tvm.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
    "efficientnet_b0": TEACHER_ARCHS["efficientnet_b0"],
}
ALL_ARCHS = {**TEACHER_ARCHS, **BASELINE_ARCHS}


def _last_linear(module):
    """(parent, attribute name, layer) of the last nn.Linear in the module tree."""
    found = None
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            found = name
    parent_name, _, attr = found.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    return parent, attr, getattr(parent, attr)


class ImageNetTeacher(nn.Module):
    """torchvision backbone whose final linear layer is replaced by a new `num_classes` head.

    `penultimate_features(x)` returns the input of that head (the pooled representation), which is the
    target of the KD feature-alignment loss in `KDTrainer`.
    """
    def __init__(self, arch, num_classes=2, pretrained=True):
        super().__init__()
        if arch not in ALL_ARCHS:
            raise ValueError(f"Unknown architecture '{arch}'. Choose from {sorted(ALL_ARCHS)}")
        ctor, weights = ALL_ARCHS[arch]
        self.arch = arch
        self.body = ctor(weights=weights if pretrained else None)
        parent, attr, last = _last_linear(self.body)
        setattr(parent, attr, nn.Identity())
        self.fc = nn.Linear(last.in_features, num_classes)

    def penultimate_features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.fc(self.body(x))


def build_teacher(arch, num_classes=2, pretrained=True):
    return ImageNetTeacher(arch, num_classes=num_classes, pretrained=pretrained)


def load_teacher(checkpoint_path, map_location="cpu"):
    """Rebuild a trained teacher from a checkpoint written by `src/train_teacher.py` (eval mode)."""
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    meta = ckpt["meta"]
    model = ImageNetTeacher(meta["arch"], num_classes=len(meta["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval(), meta
>>>>>>> Stashed changes
