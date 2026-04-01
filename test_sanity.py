import torch
import unittest
from dataset import TomographyH5Dataset
from augmentations import get_lejepa_transforms
from model import ViTEncoder, SIGReg
import os

class TestLeJEPAPipeline(unittest.TestCase):
    def setUp(self):
        self.h5_path = "/global/homes/e/elavarpa/pscratch/microct_sr_2d_project/data/processed/serpentinite_train.h5"
        
    def test_dataset_loading(self):
        if not os.path.exists(self.h5_path):
            self.skipTest(f"Data file doesn't exist: {self.h5_path}")
            
        ds = TomographyH5Dataset(
            self.h5_path,
            dataset_key=None,
            V=2,
            vmin=0.0,
            vmax=65535.0,
            is_train=True
        )
        self.assertGreater(len(ds), 0, "Dataset should have > 0 items")
        
        # Test item zero
        vs, y = ds[0]
        self.assertEqual(len(vs.shape), 4, "Should be [V, C, H, W]")
        self.assertEqual(vs.shape[0], 2, "V should be 2")
        self.assertEqual(vs.shape[1], 1, "C should be 1")
        self.assertEqual(vs.shape[2], 128, "H should be 128 due to crop")
        self.assertEqual(vs.shape[3], 128, "W should be 128 due to crop")
        
    def test_model_forward(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = ViTEncoder(proj_dim=128, img_size=128, in_chans=1).to(device)
        sigreg = SIGReg().to(device)
        
        # Batch of 4, V=2, C=1, H=128, W=128
        dummy_input = torch.randn(4, 2, 1, 128, 128).to(device)
        
        emb, proj = net(dummy_input)
        self.assertEqual(proj.shape, (2, 4, 128), "Projector shape should be (V, N, proj_dim)")
        
        loss = sigreg(proj)
        self.assertFalse(torch.isnan(loss), "Loss should not be nan")
        
if __name__ == '__main__':
    unittest.main()
