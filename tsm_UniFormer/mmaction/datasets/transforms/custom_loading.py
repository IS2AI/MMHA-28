# mmaction/datasets/transforms/custom_loading.py

import os.path as osp
import copy as cp
import mmcv
import numpy as np
from mmaction.registry import TRANSFORMS
from .loading import RawFrameDecode
from mmengine.fileio import get as mm_get


def infer_modality_from_path(frame_dir: str) -> str:
    """Heuristically decide the modality from folder name."""
    if "rgb_images" in frame_dir:
        return "RGB"
    if "depth_images" in frame_dir:
        return "Depth"
    if "depth_syn" in frame_dir:
        return "DepthSyn"
    if "thermal" in frame_dir:
        return "Thermal"
    if "event-streams" in frame_dir:
        return "Event"
    # fallback — assume RGB
    return "RGB"


@TRANSFORMS.register_module()
class CustomRawFrameDecode(RawFrameDecode):
    """Decode RGB / Depth / Thermal / Event frames."""

    def transform(self, results):
        mmcv.use_backend(self.decoding_backend)

        directory  = results["frame_dir"]
        # Always infer modality from directory, ignoring any previous value
        modality   = infer_modality_from_path(directory)

        frame_inds = np.squeeze(results["frame_inds"])
        offset     = results.get("offset", 0)
        frame_list = results.get("frame_list")

        imgs, cache = [], {}

        for frame_idx in frame_inds:
            if frame_idx in cache:
                imgs.append(cp.deepcopy(imgs[cache[frame_idx]]))
                continue
            cache[frame_idx] = len(imgs)
            frame_idx += offset

            # ------------------------------------------------------------------
            # 1. Build path
            # ------------------------------------------------------------------
            if frame_list is not None:
                if not (1 <= frame_idx <= len(frame_list)):
                    raise IndexError(f"Frame {frame_idx} out of range in {directory}")
                filename = frame_list[frame_idx - 1]
                path = osp.join(directory, filename)
            else:
                tmpl = results["filename_tmpl"]
                path = osp.join(directory, tmpl.format(frame_idx))

            if not osp.exists(path):
                raise FileNotFoundError(f"[Decode] Missing file → {path}")

            # ------------------------------------------------------------------
            # 2.  Decode by modality
            # ------------------------------------------------------------------
            if modality == "RGB":
                data_bytes = mm_get(path)
                if data_bytes is None or len(data_bytes) == 0:
                    raise RuntimeError(f"Empty bytes (RGB) → {path}")
                img = mmcv.imfrombytes(data_bytes, channel_order="rgb")
                if img is None or img.size == 0:
                    raise RuntimeError(f"OpenCV failed (RGB) → {path}")
                imgs.append(img)

            elif modality == "Depth":
                data_bytes = mm_get(path)
                img = mmcv.imfrombytes(data_bytes, flag="unchanged")
                if img is None or img.size == 0:
                    raise RuntimeError(f"OpenCV failed (Depth) → {path}")
                imgs.append(img.astype(np.float32))

            elif modality == "DepthSyn":
                data_bytes = mm_get(path)
                img = mmcv.imfrombytes(data_bytes, flag="unchanged")
                if img is None or img.size == 0:
                    raise RuntimeError(f"OpenCV failed (DepthSyn) → {path}")
                imgs.append(img.astype(np.float32))

            elif modality == "Thermal":
                data_bytes = mm_get(path)
                img = mmcv.imfrombytes(data_bytes, flag="unchanged")
                if img is None or img.size == 0:
                    raise RuntimeError(f"OpenCV failed (Thermal) → {path}")
                imgs.append(img.astype(np.float32))

            elif modality == "Event":
                arr = np.load(path)
                if arr.ndim == 2:                          # H,W → H,W,1
                    arr = arr[..., None]
                if arr.shape[2] == 1:                      # replicate
                    arr = np.repeat(arr, 3, axis=-1)
                imgs.append(arr.astype(np.float32))

            else:
                raise NotImplementedError(f"Unsupported modality {modality}")

        results["imgs"]           = imgs
        results["original_shape"] = imgs[0].shape[:2]
        results["img_shape"]      = imgs[0].shape[:2]
        results["modality"]       = modality
        return results
