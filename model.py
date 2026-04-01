import torch
import torch.nn as nn
from torchvision.ops import MLP
import timm

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128, img_size=128, in_chans=1):
        super().__init__()
        # Using a small patch setup, suitable for our 128x128 crops
        # in_chans=1 corresponds to our 1-channel grayscale micro-CT patches
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=img_size,
            in_chans=in_chans
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        # x shape: [N, V, C, H, W]
        # Flatten N, V to feed into the backbone (treating each view as independent sample)
        emb = self.backbone(x.flatten(0, 1))
        # Project and reshape back
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)
