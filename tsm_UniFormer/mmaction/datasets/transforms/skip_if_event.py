# mmaction/datasets/transforms/skip_if_event.py
from mmaction.registry import TRANSFORMS
from mmengine.config import ConfigDict
from mmcv.transforms import TRANSFORMS as MMCV_TRANSFORMS

@TRANSFORMS.register_module()
class SkipIfEvent:
    """Wrap any transform and execute it **unless** results['modality'] == 'Event'.

    Args:
        op (dict | ConfigDict): the real transform cfg.
    """
    def __init__(self, op):
        # build the real op once
        self.op = TRANSFORMS.build(op) if isinstance(op, (dict, ConfigDict)) else op

    def __call__(self, results):
        if results.get('modality') == 'Event':
            return results      # <-- no-op
        return self.op(results)  # <-- run the wrapped transform
