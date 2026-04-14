import numpy as np
from mmaction.registry import TRANSFORMS

# Thermal normalization values from per-modality statistics
THERMAL_MEAN = np.array([0.483, 0.027, 0.508])
THERMAL_STD = np.array([0.133, 0.106, 0.194])


@TRANSFORMS.register_module()
class NormalizeThermal:
    """Normalize thermal images using per-modality statistics."""
    
    def __init__(self, to_float32=True):
        self.to_float32 = to_float32
        
    def __call__(self, results):
        imgs = results['imgs']  # list of (C, H, W) numpy arrays
        
        # Reshape mean/std to broadcast with img
        extra_dims = (1,) * (imgs[0].ndim - 1)  # e.g. (1,1) for (C,H,W)
        mean = THERMAL_MEAN.reshape((3,) + extra_dims)
        std = THERMAL_STD.reshape((3,) + extra_dims)
        
        proc = []
        for img in imgs:
            # Ensure images are float32 before doing arithmetic
            if self.to_float32 and img.dtype != np.float32:
                img = img.astype(np.float32)
            
            # Convert to [0,1] range
            img = img / 255.0
            
            # Subtract mean and divide by std
            img = (img - mean) / std
            
            # Cast explicitly back to float32
            img = img.astype(np.float32)
            proc.append(img)
        
        results['imgs'] = np.stack(proc, axis=0)
        return results 