# mmaction/datasets/custom_event_dataset.py
import os
from mmaction.registry import DATASETS
from .custom_multimodal_dataset import CustomMultimodalDataset


@DATASETS.register_module()
class CustomEventDataset(CustomMultimodalDataset):
    """Load only event stream samples from combined multimodal annotation files."""

    def load_data_list(self):
        # Get all data from parent class
        data_list = super().load_data_list()
        
        # Filter to only keep event stream samples
        event_data_list = []
        for vid in data_list:
            fdir = vid["frame_dir"]
            if "event-streams" in fdir:
                event_data_list.append(vid)
        
        print(f"Filtered {len(data_list)} total samples to {len(event_data_list)} event stream samples")
        return event_data_list 