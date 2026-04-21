import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from model import ViTEncoder
from dataset import TomographyH5Dataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

def visualize_pca(checkpoint_path, h5_path, output_dir="viz_pca"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration (matching training)
    cfg = OmegaConf.create({
        "proj_dim": 128, "img_size": 512, "in_chans": 1,
        "bs": 1, "V": 1, "vmin": 0.0, "vmax": 65535.0
    })
    
    # Load model
    model = ViTEncoder(proj_dim=cfg.proj_dim, img_size=cfg.img_size, in_chans=cfg.in_chans).to("cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Handle both wrapped checkpoints and raw state dicts
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Handle DDP/Compile state dict (remove 'module.' or '_orig_mod.' prefixes)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "").replace("module.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Load dataset
    ds = TomographyH5Dataset(h5_path, vmin=cfg.vmin, vmax=cfg.vmax, is_train=False)
    # Use a fixed seed for the generator to ensure the same images are chosen across different epoch runs
    g = torch.Generator()
    g.manual_seed(42)
    loader = DataLoader(ds, batch_size=1, shuffle=True, generator=g)
    
    # Process a few samples
    with torch.no_grad():
        for i, (img, _) in enumerate(loader):
            if i >= 5: break # Visualize 5 samples
            
            img = img.to("cpu")
            # Extract features from the backbone
            # vit_small_patch8_224 -> grid is 64x64 for 512x512 input
            features = model.backbone.forward_features(img)
            
            # Remove [CLS] token (index 0)
            patch_tokens = features[:, 1:, :] # [1, 4096, 384]
            patch_tokens = patch_tokens.squeeze(0) # [4096, 384]
            
            # Normalize tokens
            patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)
            
            # PCA using SVD
            u, s, v = torch.pca_lowrank(patch_tokens, q=3)
            
            # Print explained variance ratio (approximate using squared singular values)
            exp_var = (s**2) / (patch_tokens.var(dim=0).sum() * (patch_tokens.size(0)-1))
            print(f"\n--- Slice {i} PCA Statistics ---")
            print(f"Explained Variance (Approx): PC1: {exp_var[0]:.2%}, PC2: {exp_var[1]:.2%}, PC3: {exp_var[2]:.2%}")
            
            # Project tokens onto the first 3 PCs
            pcs = patch_tokens @ v[:, :3] # [4096, 3]
            
            # Print sample projected values for the center patch
            center_pc = pcs[2048] # Middle of 64x64 grid
            print(f"Sample Patch (Center) 3D Projection: {center_pc.tolist()}")
            
            # Reshape tokens to [C, H, W] for upsampling
            pcs = pcs.reshape(64, 64, 3).permute(2, 0, 1).unsqueeze(0) # [1, 3, 64, 64]
            
            # Upsample to full image size (512x512)
            import torch.nn.functional as F
            pca_img = F.interpolate(pcs, size=(512, 512), mode='bilinear', align_corners=False)
            pca_img = pca_img.squeeze(0).permute(1, 2, 0).cpu().numpy() # [512, 512, 3]
            
            # Robust normalization (clipping extremes)
            for c in range(3):
                low, high = np.percentile(pca_img[..., c], [1, 99])
                pca_img[..., c] = np.clip((pca_img[..., c] - low) / (high - low), 0, 1)
            
            # Plot a comprehensive view
            orig_img = img.squeeze().cpu().numpy()
            plt.figure(figsize=(25, 5))
            
            # Original
            plt.subplot(1, 5, 1)
            plt.imshow(orig_img, cmap='gray')
            plt.title("Original Slice")
            plt.axis('off')
            
            # RGB Composite
            plt.subplot(1, 5, 2)
            plt.imshow(pca_img)
            plt.title("PCA (RGB Composite)")
            plt.axis('off')
            
            # Individual PCs
            for j in range(3):
                plt.subplot(1, 5, 3 + j)
                pc_map = pca_img[..., j]
                plt.imshow(pc_map, cmap='viridis')
                plt.title(f"PC {j+1}")
                plt.axis('off')
            
            # Use a naming convention that includes the epoch if provided
            epoch_str = os.path.basename(checkpoint_path).replace(".pth", "")
            out_file = os.path.join(output_dir, f"{epoch_str}_slice_{i}.png")
            plt.savefig(out_file, bbox_inches='tight')
            plt.close()
            print(f"Saved detailed PCA visualization to {out_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/lejepa_epoch_6.pth")
    parser.add_argument("--h5", default="/global/homes/e/elavarpa/pscratch/microct_sr_2d_project/data/processed/serpentinite_train.h5")
    args = parser.parse_args()
    
    visualize_pca(args.ckpt, args.h5)
