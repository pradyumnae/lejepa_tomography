import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import wandb
import hydra
from omegaconf import DictConfig
import tqdm

from dataset import TomographyH5Dataset
from model import ViTEncoder, SIGReg

@hydra.main(version_base=None, config_name="config")
def main(cfg: DictConfig):
    # Initialize wandb (dryrun if not logging)
    wandb.init(project="LeJEPA_Tomography", mode="disabled" if cfg.debug else "online", config=dict(cfg))
    torch.manual_seed(0)

    # Initialize micro-CT 32-bit dataloader
    # Replace with path to actual processed serpentinite_train.h5
    train_ds = TomographyH5Dataset(
        h5_path="data/processed/serpentinite_train.h5", 
        dataset_key=cfg.dataset_key,
        V=cfg.V,
        vmin=cfg.vmin,
        vmax=cfg.vmax,
        is_train=True
    )
    
    train = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=cfg.num_workers
    )

    # LeJEPA model components
    net = ViTEncoder(proj_dim=cfg.proj_dim, img_size=cfg.img_size, in_chans=1).to("cuda")
    
    # Optional downstream probe (not strictly needed since unsupervised pretraining, 
    # but kept as in MINIMAL.md for tracking potential dummy tasks if needed)
    probe = torch.nn.Sequential(torch.nn.LayerNorm(512), torch.nn.Linear(512, 10)).to("cuda")
    sigreg = SIGReg().to("cuda")

    # Optimizer and scheduler configuration
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    
    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=max(1, warmup_steps))
    s2 = CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps), eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=torch.cuda.is_available())

    # The user requested NOT to actually train yet, so we just dry-run initialize
    print("Project structure initialized successfully. Model instantiated.")
    print(f"Dataset length: {len(train_ds)}")
    print(f"Sample dataloader loop ready. Skipping training loop for now.")
    
    # Basic dummy dry-run to ensure shapes compile
    try:
        vs, y = next(iter(train))
        print(f"Data batch shape: {vs.shape}")
        vs = vs.to("cuda")
        emb, proj = net(vs)
        print(f"Encoder embeddings shape: {emb.shape}")
        print(f"Projector output shape: {proj.shape}")
        sigreg_loss = sigreg(proj)
        print("Model forwards pass successful!")
    except Exception as e:
        print(f"Dry-run initialization error or data not found: {str(e)}")

    wandb.finish()

if __name__ == "__main__":
    # Provides defaults in case we just run `python main.py` directly
    # Need to run with default configs via hydra kwargs from terminal or yaml
    from omegaconf import OmegaConf
    default_cfg = OmegaConf.create({
        "debug": True,
        "dataset_key": "data",
        "V": 2,
        "vmin": 0.0,
        "vmax": 65535.0, # Approximate for 16/32bit
        "bs": 8,
        "num_workers": 2,
        "proj_dim": 128,
        "img_size": 128,
        "lr": 1e-4,
        "epochs": 1,
        "lamb": 0.5
    })
    
    # Manually pass the config here since we don't have a config.yaml for hydra yet
    # We cheat the hydra decorator for the dry run.
    main(default_cfg)
