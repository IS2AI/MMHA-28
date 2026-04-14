import numpy as np
from mmaction.registry import TRANSFORMS

# Event normalization values from per-modality statistics
EVENT_MEAN = np.array([0.0, 0.0, 0.0])
EVENT_STD = np.array([1.0, 1.0, 1.0])


@TRANSFORMS.register_module()
class NormalizeEvent:
    """Normalize event stream data using per-modality statistics."""
    
    def __init__(self, to_float32=True):
        self.to_float32 = to_float32
        
    def __call__(self, results):
        imgs = results['imgs']  # list of (C, H, W) numpy arrays
        
        # Reshape mean/std to broadcast with img
        extra_dims = (1,) * (imgs[0].ndim - 1)  # e.g. (1,1) for (C,H,W)
        mean = EVENT_MEAN.reshape((3,) + extra_dims)
        std = EVENT_STD.reshape((3,) + extra_dims)
        
        proc = []
        for img in imgs:
            # Ensure images are float32 before doing arithmetic
            if self.to_float32 and img.dtype != np.float32:
                img = img.astype(np.float32)
            
            # For event streams, we don't divide by 255.0 (they're already in appropriate range)
            # This matches the behavior in normalize_per_modality.py
            
            # Subtract mean and divide by std (essentially no-op for events: mean=0, std=1)
            img = (img - mean) / std
            
            # Cast explicitly back to float32
            img = img.astype(np.float32)
            proc.append(img)
        
        results['imgs'] = np.stack(proc, axis=0)
        return results 