# mmaction/datasets/custom_depth_dataset.py
import os
from mmaction.registry import DATASETS
from .custom_multimodal_dataset import CustomMultimodalDataset


@DATASETS.register_module()
class CustomDepthDataset(CustomMultimodalDataset):
    """Load only depth samples from combined multimodal annotation files."""

    def load_data_list(self):
        # Get all data from parent class
        data_list = super().load_data_list()
        
        # Filter to only keep depth samples
        depth_data_list = []
        for vid in data_list:
            fdir = vid["frame_dir"]
            if "depth_images" in fdir:
                depth_data_list.append(vid)
        
        print(f"Filtered {len(data_list)} total samples to {len(depth_data_list)} depth samples")
        return depth_data_list 