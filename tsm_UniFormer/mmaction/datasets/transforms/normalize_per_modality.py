# mmaction/datasets/transforms/normalize_per_modality.py

import numpy as np
from mmaction.registry import TRANSFORMS

# ---------------------------------------------------------------------------- #
#  Per-modality mean/std values.
# ---------------------------------------------------------------------------- #
MEAN_STD = {
    'RGB':     (
        np.array([0.485, 0.456, 0.406]),
        np.array([0.229, 0.224, 0.225])
    ),
    'Depth':   (
        np.array([0.591, 0.278, 0.258]),
        np.array([0.463, 0.355, 0.324])
    ),
    'DepthSyn': (
        np.array([0.591, 0.278, 0.258]),
        np.array([0.463, 0.355, 0.324])
    ),
    'Thermal': (
        np.array([0.483, 0.027, 0.508]),
        np.array([0.133, 0.106, 0.194])
    ),
    'Event':   (
        np.array([ 0.0,  0.0,  0.0 ]),
        np.array([ 1.0,  1.0,  1.0 ])
    ),
}

@TRANSFORMS.register_module()
class NormalizePerModality:
    """Subtract mean / divide std based on results['modality'], ensure float32 output."""
    def __init__(self, to_float32=True):
        self.to_float32 = to_float32

    def __call__(self, results):
        imgs     = results['imgs']                    # list of (C, H, W) numpy arrays
        modality = results['modality']
        mean, std = MEAN_STD.get(modality, MEAN_STD['RGB'])

        # Reshape mean/std to broadcast with img. Works for (C,H,W) or (C,T,H,W)
        extra_dims = (1,) * (imgs[0].ndim - 1)  # e.g. (1,1) or (1,1,1)
        mean = mean.reshape((3,) + extra_dims)
        std  = std.reshape((3,) + extra_dims)

        proc = []
        for img in imgs:
            # Ensure images are float32 before doing arithmetic
            if self.to_float32 and img.dtype != np.float32:
                img = img.astype(np.float32)
            if modality != 'Event':
                img = img / 255.0

            # Subtract the per-channel mean and divide by per-channel std
            img = (img - mean) / std

            # Cast explicitly back to float32
            img = img.astype(np.float32)
            proc.append(img)

        results['imgs'] = np.stack(proc, axis=0)
        return results
