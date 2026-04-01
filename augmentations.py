import torch
import torch.nn as nn
from torchvision.transforms import v2
import numpy as np

class CustomIntensityWindowing(nn.Module):
    """
    Clips 32-bit intensity values and normalizes them to [0, 1].
    Given a min and max value typical for the physical 32-bit tomogram.
    """
    def __init__(self, vmin, vmax):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax

    def forward(self, img):
        # img is expected to be a float32 tensor
        img_clipped = torch.clamp(img, self.vmin, self.vmax)
        img_norm = (img_clipped - self.vmin) / (self.vmax - self.vmin)
        return img_norm

class RandomGaussianNoise(nn.Module):
    """
    Adds Gaussian sensor noise.
    """
    def __init__(self, std_range=(0.01, 0.05)):
        super().__init__()
        self.std_range = std_range

    def forward(self, img):
        std = torch.empty(1).uniform_(self.std_range[0], self.std_range[1]).item()
        noise = torch.randn_like(img) * std
        return torch.clamp(img + noise, 0, 1)

class RandomRingArtifact(nn.Module):
    """
    Simulates concentric ring artifacts commonly found in tomography.
    """
    def __init__(self, max_rings=3, max_intensity=0.1):
        super().__init__()
        self.max_rings = max_rings
        self.max_intensity = max_intensity

    def forward(self, img):
        _, h, w = img.shape
        center_y, center_x = h // 2, w // 2
        
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        r = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
        
        artifact_mask = torch.zeros_like(r)
        
        num_rings = torch.randint(1, self.max_rings + 1, (1,)).item()
        for _ in range(num_rings):
            radius = torch.empty(1).uniform_(10, min(h, w) // 2).item()
            width = torch.empty(1).uniform_(1, 3).item()
            intensity = torch.empty(1).uniform_(0.02, self.max_intensity).item()
            
            # Create ring
            ring = torch.exp(-0.5 * ((r - radius) / width) ** 2)
            artifact_mask += ring * intensity
            
        return torch.clamp(img + artifact_mask.unsqueeze(0), 0, 1)

def get_lejepa_transforms(vmin=0.0, vmax=65535.0, is_target=False):
    """
    Returns the transformation pipeline for the LeJEPA context or target views
    for 32-bit grayscale tomography data.
    """
    # Adjust vmin/vmax defaults depending on actual 32-bit dataset values
    base_transforms = [
        CustomIntensityWindowing(vmin=vmin, vmax=vmax),
        # Assuming input is already float tensor [1, H, W]
    ]
    
    if not is_target:
        # Context view gets heavier augmentations
        aug_transforms = [
            v2.RandomResizedCrop(128, scale=(0.2, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([RandomGaussianNoise()], p=0.5),
            v2.RandomApply([RandomRingArtifact()], p=0.3),
        ]
    else:
        # Target view gets lighter augmentations (or just cropped)
        aug_transforms = [
            v2.RandomResizedCrop(128, scale=(0.5, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ]
        
    return v2.Compose(base_transforms + aug_transforms)
