import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
    # Initialize DDP
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    num_gpus = torch.cuda.device_count()
    print(f"[Rank {global_rank}/{world_size} - Local {local_rank}] Available GPUs: {num_gpus}")

    if torch.cuda.is_available():
        if local_rank < num_gpus:
            torch.cuda.set_device(local_rank)
        else:
            print(f"Warning: local_rank {local_rank} >= num_gpus {num_gpus}. Defaulting to 0.")
            torch.cuda.set_device(0)
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Initialize wandb (only on main rank)
    if global_rank == 0:
        wandb.init(project="LeJEPA_Tomography", mode="disabled" if cfg.debug else "online", config=dict(cfg))
    
    torch.manual_seed(0)

    train_ds = TomographyH5Dataset(
        h5_path="/global/homes/e/elavarpa/pscratch/microct_sr_2d_project/data/processed/serpentinite_train.h5", 
        dataset_key=cfg.dataset_key,
        V=cfg.V, vmin=cfg.vmin, vmax=cfg.vmax, is_train=True
    )
    
    sampler = DistributedSampler(train_ds, shuffle=True)
    train = DataLoader(train_ds, batch_size=cfg.bs, sampler=sampler, drop_last=True, num_workers=cfg.num_workers)

    net = ViTEncoder(proj_dim=cfg.proj_dim, img_size=cfg.img_size, in_chans=1).to(local_rank)
    net = DDP(net, device_ids=[local_rank], find_unused_parameters=True)
    
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 10)).to(local_rank)
    probe = DDP(probe, device_ids=[local_rank])
    sigreg = SIGReg().to(local_rank)

    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    
    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=max(1, warmup_steps))
    s2 = CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps), eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Training Loop
    if global_rank == 0:
        print("Starting distributed training...")
        
    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        net.train(), probe.train()
        
        # Only show progress bar on main rank
        pbar = tqdm.tqdm(train, total=len(train)) if global_rank == 0 else train
        
        for vs, y in pbar:
            vs = vs.to(local_rank, non_blocking=True)
            y = y.to(local_rank, non_blocking=True)
            
            with autocast("cuda", dtype=torch.bfloat16):
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                y_rep, yhat = y.repeat_interleave(cfg.V), probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y_rep)
                loss = lejepa_loss + probe_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            
            if global_rank == 0:
                wandb.log({
                    "train/probe": probe_loss.item(),
                    "train/lejepa": lejepa_loss.item(),
                    "train/sigreg": sigreg_loss.item(),
                    "train/inv": inv_loss.item(),
                })

    if global_rank == 0:
        wandb.finish()

if __name__ == "__main__":
    from omegaconf import OmegaConf
    default_cfg = OmegaConf.create({
        "debug": True,
        "dataset_key": "data",
        "V": 2, "vmin": 0.0, "vmax": 65535.0,
        "bs": 16, # Increased batch size for 4 gpus
        "num_workers": 4,
        "proj_dim": 128, "img_size": 128,
        "lr": 1e-4, "epochs": 5, "lamb": 0.5
    })
    
    # Normally hydra handles sys args, passing defaults directly here acts as dry-run fallback if no yaml provided
    main(default_cfg)
