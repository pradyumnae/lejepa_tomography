import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from augmentations import get_lejepa_transforms

class TomographyH5Dataset(Dataset):
    """
    A PyTorch Dataset for reading 32-bit tomography data from HDF5.
    Returns V views, where typcally elements are transformed differently 
    (e.g., target vs context views for LeJEPA).
    """
    def __init__(self, h5_path, dataset_key='data', V=2, 
                 vmin=0.0, vmax=1.0, is_train=True):
        self.h5_path = h5_path
        self.dataset_key = dataset_key
        self.V = V
        self.is_train = is_train
        
        # Open quickly to check length (assuming [N, H, W] or [H, W, D])
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f[self.dataset_key].shape[0]

        # For LeJEPA, typically V contexts. If we want a specific target transform,
        # we can define diverse pipelines here.
        # Here we just use context view transforms for all V views for simplicity,
        # or we could make the first one the target.
        self.transform_target = get_lejepa_transforms(vmin, vmax, is_target=True)
        self.transform_context = get_lejepa_transforms(vmin, vmax, is_target=False)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        # Open per worker (HDF5 doesn't like shared file handles across workers)
        with h5py.File(self.h5_path, 'r') as f:
            # Read single slice as basic example
            slice_data = f[self.dataset_key][i]
            
        # Convert to float32 tensor [1, H, W]
        img_tensor = torch.from_numpy(slice_data.astype(np.float32)).unsqueeze(0)
        
        # Apply LeJEPA multi-view augmentations
        if self.is_train:
            # e.g., view 0 is target, everything else is context
            views = [self.transform_target(img_tensor)]
            for _ in range(1, self.V):
                views.append(self.transform_context(img_tensor))
            return torch.stack(views), 0 # Dummy label for unsupervised
        else:
            # Validation just center crops or standardizes
            return self.transform_target(img_tensor), 0

