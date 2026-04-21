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
    if global_rank == 0:
        print(f"World Size: {world_size}")
        print(f"GPUs per task: {num_gpus}")

    if torch.cuda.is_available():
        # Using local_rank as the device index now that we are using torchrun
        # with multiple GPUs per Slurm task.
        torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Initialize wandb (only on main rank)
    if global_rank == 0:
        wandb.init(project="LeJEPA_Tomography", mode="disabled" if cfg.debug else "online", config=dict(cfg))
    
    # Enable TF32 for significant speedup on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    torch.manual_seed(0)

    train_ds = TomographyH5Dataset(
        h5_path="/global/homes/e/elavarpa/pscratch/microct_sr_2d_project/data/processed/serpentinite_train.h5", 
        dataset_key=cfg.dataset_key, # If None, it will discover all scans
        V=cfg.V, vmin=cfg.vmin, vmax=cfg.vmax, is_train=True
    )
    
    sampler = DistributedSampler(train_ds, shuffle=True)
    train = DataLoader(
        train_ds, batch_size=cfg.bs, sampler=sampler, drop_last=True, 
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True
    )

    net = ViTEncoder(proj_dim=cfg.proj_dim, img_size=cfg.img_size, in_chans=1).to("cuda")
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # Auto-resume logic (Load BEFORE compile/DDP to avoid key mismatches)
    start_epoch = 0
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_files = [f for f in os.listdir("checkpoints") if f.startswith("lejepa_epoch_") and f.endswith(".pth")]
    if ckpt_files:
        epochs_found = [int(f.split("_")[-1].split(".")[0]) for f in ckpt_files]
        latest_epoch = max(epochs_found)
        ckpt_path = f"checkpoints/lejepa_epoch_{latest_epoch}.pth"
        if global_rank == 0:
            print(f"Loading checkpoint: {ckpt_path}", flush=True)
        
        checkpoint = torch.load(ckpt_path, map_location="cuda")
        
        # Strip prefixes from state_dict keys (e.g., _orig_mod. or module.)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("_orig_mod.", "").replace("module.", "")
            new_state_dict[name] = v
            
        net.load_state_dict(new_state_dict)
        # We will load optimizer/scheduler/start_epoch later after they are initialized
    else:
        checkpoint = None

    net = torch.compile(net) # Compile entire backbone for 20-30% speedup
    net = DDP(net, device_ids=[local_rank], find_unused_parameters=False)
    
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 10)).to("cuda")
    probe = DDP(probe, device_ids=[local_rank])
    sigreg = SIGReg().to("cuda")

    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    
    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=max(1, warmup_steps))
    s2 = CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps), eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])
    
    if checkpoint is not None:
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if global_rank == 0:
            print(f"Resuming from epoch {start_epoch}", flush=True)

    # Training Loop
    print(f"Starting distributed training on {len(train_ds)} slices...", flush=True)
    for epoch in range(start_epoch, cfg.epochs):
        sampler.set_epoch(epoch)
        net.train(), probe.train()
        
        # Only show progress bar on main rank
        pbar = tqdm.tqdm(train, total=len(train)) if global_rank == 0 else train
        
        for i, (vs, y) in enumerate(pbar):
            if global_rank == 0 and i == 0:
                print(f"Successfully started Batch 0 of Epoch {epoch}", flush=True)
            
            vs = vs.to("cuda", non_blocking=True)
            y = y.to("cuda", non_blocking=True)
            
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
        
        # Save checkpoint at end of epoch
        if global_rank == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/lejepa_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.module.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss.item(),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}", flush=True)

    if global_rank == 0:
        torch.save(net.module.state_dict(), "checkpoints/lejepa_final.pth")
        wandb.finish()

if __name__ == "__main__":
    from omegaconf import OmegaConf
    default_cfg = OmegaConf.create({
        "debug": True,
        "dataset_key": None, # Discover all scans
        "V": 2, "vmin": 0.0, "vmax": 65535.0,
        "bs": 8, # Per GPU batch size
        "num_workers": 4,
        "proj_dim": 128, "img_size": 512,
        "lr": 1e-4, "epochs": 30, "lamb": 0.5
    })
    
    main(default_cfg)
