# mmaction/datasets/custom_rgb_dataset.py
import os
from mmaction.registry import DATASETS
from .custom_multimodal_dataset import CustomMultimodalDataset


@DATASETS.register_module()
class CustomRGBDataset(CustomMultimodalDataset):
    """Load only RGB samples from combined multimodal annotation files."""

    def load_data_list(self):
        # Get all data from parent class
        data_list = super().load_data_list()
        
        # Filter to only keep RGB samples
        rgb_data_list = []
        for vid in data_list:
            fdir = vid["frame_dir"]
            if "rgb_images" in fdir:
                rgb_data_list.append(vid)
        
        print(f"Filtered {len(data_list)} total samples to {len(rgb_data_list)} RGB samples")
        return rgb_data_list 