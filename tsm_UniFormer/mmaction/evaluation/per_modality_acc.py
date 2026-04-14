# per_modality_acc_metric.py
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict, OrderedDict
import copy

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmaction.registry import METRICS
from mmaction.evaluation import (top_k_accuracy, mean_class_accuracy)

@METRICS.register_module()
class PerModalityAccuracy(BaseMetric):
    """Clip-level top-1 (and optional top-5) accuracy **per modality**.

    It re-uses the same logic as AccMetric, but buckets clips by
    `data_batch['modality'][i]`.  Because it never tries to fuse clips
    into videos, its numbers match `AccMetric` exactly whenever a run
    contains just one modality.
    """

    default_prefix: Optional[str] = 'acc'

    def __init__(self,
                 top_k: Tuple[int, ...] = (1, 5),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.top_k = top_k
        # internal storage: modality → list of (pred_np, label_int)
        self._store = defaultdict(list)

    # ------------------------------------------------------------------
    # 1. accumulate clip-level predictions
    # ------------------------------------------------------------------
    def process(self,
                data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        modalities = data_batch['modality']   # list[str] with same length
        for i, ds in enumerate(data_samples):
            modality = modalities[i]

            # ---- extract numpy score vector ----
            pred = ds['pred_score']
            if isinstance(pred, dict):
                raise NotImplementedError(
                    "PerModalityAccMetric doesn't support dict-predictions "
                    "(e.g. RGBPoseConv3D).")
            pred_np = pred.detach().cpu().numpy()

            # ---- extract ground-truth label ----
            label = int(ds['gt_label'])   # tensor or int → int

            self._store[modality].append((pred_np, label))

    # ------------------------------------------------------------------
    # 2. compute per-modality accuracies
    # ------------------------------------------------------------------
    def compute_metrics(self, results=None) -> Dict:
        out = OrderedDict()
        for modality, pairs in self._store.items():
            if not pairs:   # safety
                continue
            preds, labels = zip(*pairs)
            preds = list(preds)
            labels = list(labels)

            # --- top-k ---
            topk = top_k_accuracy(preds, labels, self.top_k)
            for k, acc in zip(self.top_k, topk):
                out[f'{modality}_acc/top{k}'] = acc

            # --- mean class accuracy (optional) ---
            mca = mean_class_accuracy(preds, labels)
            out[f'{modality}_acc/mean1'] = mca

        # clear for next epoch
        self._store.clear()
        return out
