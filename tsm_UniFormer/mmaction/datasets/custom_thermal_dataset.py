# mmaction/datasets/custom_thermal_dataset.py
import os
from mmaction.registry import DATASETS
from .custom_multimodal_dataset import CustomMultimodalDataset


@DATASETS.register_module()
class CustomThermalDataset(CustomMultimodalDataset):
    """Load only thermal samples from combined multimodal annotation files."""

    def load_data_list(self):
        # Get all data from parent class
        data_list = super().load_data_list()
        
        # Filter to only keep thermal samples
        thermal_data_list = []
        for vid in data_list:
            fdir = vid["frame_dir"]
            if "thermal" in fdir:
                thermal_data_list.append(vid)
        
        print(f"Filtered {len(data_list)} total samples to {len(thermal_data_list)} thermal samples")
        return thermal_data_list 