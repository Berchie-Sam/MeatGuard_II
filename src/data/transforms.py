import random
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance

# Constant benchmark values
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

class RandomGaussianBlur:
    """Random Gaussian blur to simulate out-of-focus market shots."""
    def __init__(self, p=0.3, radius_range=(0.1, 2.0)):
        self.p = p
        self.r = radius_range

    def __call__(self, img):
        if random.random() < self.p:
            r = random.uniform(*self.r)
            return transforms.GaussianBlur(kernel_size=5, sigma=r)(img)
        return img

class RandomShadow:
    """Simulate a shadow strip — stays in PIL space."""
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        x1 = random.randint(0, w // 2)
        x2 = random.randint(w // 2, w)
        factor = random.uniform(0.3, 0.7)
        region = img.crop((x1, 0, x2, h))
        region = ImageEnhance.Brightness(region).enhance(factor)
        out = img.copy()
        out.paste(region, (x1, 0))
        return out

class RandomFog:
    """Add a white fog overlay — stays in PIL space."""
    def __init__(self, p=0.2, intensity_range=(0.05, 0.25)):
        self.p = p
        self.i = intensity_range

    def __call__(self, img):
        if random.random() > self.p:
            return img
        alpha = random.uniform(*self.i)
        fog = Image.new("RGB", img.size, (255, 255, 255))
        return Image.blend(img.convert("RGB"), fog, alpha)

def build_train_transform(strategy: str) -> transforms.Compose:
    """
    Returns the augmentation pipeline based on the benchmark strategy.
    """
    norm = transforms.Normalize(mean=MEAN, std=STD)
    to_t = transforms.ToTensor()

    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))

    if strategy == "baseline":
        return transforms.Compose([
            resize,
            transforms.CenterCrop(IMG_SIZE),
            to_t, norm,
        ])

    elif strategy == "standard":
        return transforms.Compose([
            resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            to_t, norm,
        ])

    elif strategy == "autoaugment":
        return transforms.Compose([
            resize,
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
            to_t, norm,
        ])

    elif strategy == "randaugment":
        return transforms.Compose([
            resize,
            transforms.RandAugment(num_ops=2, magnitude=9),
            to_t, norm,
        ])

    elif strategy == "cutmix_mixup":
        # CutMix/MixUp are applied in the training loop, not the transform pipeline
        return transforms.Compose([
            resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            to_t, norm,
        ])

    elif strategy == "domain":
        return transforms.Compose([
            resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            RandomGaussianBlur(p=0.3),
            RandomShadow(p=0.3),
            RandomFog(p=0.2),
            to_t, norm,
        ])

    else:
        raise ValueError(f"Unknown augmentation strategy: {strategy}")

def get_val_test_transform():
    """Standard clean transform for evaluation."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
