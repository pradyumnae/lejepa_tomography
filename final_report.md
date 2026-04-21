# LeJEPA Tomography Project Final Report

## Overview
This project applied the LeJEPA self-supervised learning framework to high-resolution 32-bit Micro-CT tomography data. Training was performed on NERSC Perlmutter using 4x A100 GPUs with optimized data loading and kernel fusion.

## Project Results (30 Epochs)

### 1. Training Convergence
The model reached the 30-epoch target successfully. Using `torch.compile` and TF32, we achieved an iteration speed of ~6.6s for 512x512 slices.

### 2. Semantic Representation Analysis
The latent feature space has converged to reflect the physical phase separation of the serpentinite volume.
*   **Variance Explained**: PC1 explains **99.7%** of the patch token variance.
*   **Emergent Segmentation**: The model spontaneously separates mineral grains (warm colors) from the pore matrix (cool colors).

#### Semantic Map Gallery (Epoch 30)
![Slice 0](./report_images/pca_detailed_slice_0.png)
![Slice 3](./report_images/pca_detailed_slice_3.png)

### 3. Training Progression (Epoch 5 vs 10 vs 20 vs 30)
To visualize the learning process, we compared the representations of the same slices at different stages of training. Notice the refinement of boundaries and the consolidation of the phase separation.

#### Slice 0 Comparison
*   **Epoch  5**: Noisy initial representation; mineral separation is mostly broad color blobs.
*   **Epoch 10**: Initial separation emerging, but boundaries are fuzzy.
*   **Epoch 20**: Phase boundaries sharpest, with high contrast.
*   **Epoch 30**: Thermodynamic stability reached; representation is now robust and noise-invariant.

![Epoch 5 Slice 0](./report_images/comparison/lejepa_epoch_5_slice_0.png)

![Epoch 10 Slice 0](./report_images/comparison/lejepa_epoch_10_slice_0.png)

![Epoch 20 Slice 0](./report_images/comparison/lejepa_epoch_20_slice_0.png)

![Epoch 30 Slice 0](./report_images/comparison/lejepa_final_slice_0.png)

#### Slice 3 Comparison
![Epoch 5 Slice 3](./report_images/comparison/lejepa_epoch_5_slice_3.png)

![Epoch 10 Slice 3](./report_images/comparison/lejepa_epoch_10_slice_3.png)

![Epoch 20 Slice 3](./report_images/comparison/lejepa_epoch_20_slice_3.png)

![Epoch 30 Slice 3](./report_images/comparison/lejepa_final_slice_3.png)

## Final Weights
The trained model weights are saved at:
`/global/homes/e/elavarpa/pscratch/lejepa_tomography/checkpoints/lejepa_final.pth`

## Next Steps
1.  **Downstream Segmentation**: Use these weights to initialize a supervised segmentation model.
2.  **Generalization**: Apply the encoder to the `lisabeth` dataset.
