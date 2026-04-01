import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from augmentations import get_lejepa_transforms

class TomographyH5Dataset(Dataset):
    """
    Treats multiple HDF5 datasets (scans) as a single large continuous volume.
    Automatically discovers scans if dataset_key is None.
    """
    def __init__(self, h5_path, dataset_key=None, V=2, 
                 vmin=0.0, vmax=65535.0, is_train=True):
        self.h5_path = h5_path
        self.V = V
        self.is_train = is_train
        self.vmin = vmin
        self.vmax = vmax
        
        with h5py.File(self.h5_path, 'r') as f:
            if dataset_key is None:
                self.dataset_names = sorted(list(f.keys()))
            else:
                self.dataset_names = [dataset_key] if isinstance(dataset_key, str) else dataset_key

            self.scan_infos = []
            self.total_len = 0
            
            for name in self.dataset_names:
                dset = f[name]
                # Assuming shape [D, H, W]
                d, h, w = dset.shape
                self.scan_infos.append({
                    'name': name,
                    'start_idx': self.total_len,
                    'end_idx': self.total_len + d,
                    'shape': (d, h, w)
                })
                self.total_len += d

        self.transform_target = get_lejepa_transforms(vmin, vmax, is_target=True)
        self.transform_context = get_lejepa_transforms(vmin, vmax, is_target=False)
        self.h5_file = None

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        # Find which scan this index belongs to
        target_info = None
        for info in self.scan_infos:
            if info['start_idx'] <= idx < info['end_idx']:
                target_info = info
                break
        
        if target_info is None:
            raise IndexError("Global index out of range")

        local_idx = idx - target_info['start_idx']
        
        # Read the slice
        # Some scans might be stored differently, but we assume [D, H, W]
        slice_data = self.h5_file[target_info['name']][local_idx]
            
        # Convert to float32 tensor [1, H, W]
        img_tensor = torch.from_numpy(slice_data.astype(np.float32)).unsqueeze(0)
        
        # Apply LeJEPA multi-view augmentations
        if self.is_train:
            views = [self.transform_target(img_tensor)]
            for _ in range(1, self.V):
                views.append(self.transform_context(img_tensor))
            return torch.stack(views), 0 
        else:
            return self.transform_target(img_tensor), 0
