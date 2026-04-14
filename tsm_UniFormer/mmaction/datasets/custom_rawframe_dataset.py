import os
from mmaction.registry import DATASETS
from .rawframe_dataset import RawframeDataset


@DATASETS.register_module()
class CustomMultimodalDataset(RawframeDataset):
    """Load RGB / Depth / Thermal / Event frame folders."""

    def load_data_list(self):
        data_list = super().load_data_list()

        for vid in data_list:
            fdir = vid["frame_dir"]
            if not os.path.isdir(fdir):
                raise FileNotFoundError(f"Directory not found: {fdir}")

            # Decide modality + prefix/ext
            if "rgb_images" in fdir:
                prefix, ext, modality = "rgb_",  ".png", "RGB"
            elif "depth_images" in fdir:
                prefix, ext, modality = "depth_", ".png", "Depth"
            elif "thermal" in fdir:
                prefix, ext, modality = "frame_", ".jpg", "Thermal"
            elif "event-streams" in fdir:
                prefix, ext, modality = "frame_", ".npy", "Event"
            else:
                raise RuntimeError(f"Cannot infer modality for {fdir}")

            files = [fn for fn in os.listdir(fdir)
                     if fn.startswith(prefix) and fn.endswith(ext)]
            if not files:
                raise FileNotFoundError(
                    f"No '{prefix}*{ext}' files in {fdir}"
                )

            # sort numerically
            def key(fn):
                stem = os.path.splitext(fn)[0]
                return int(stem[len(prefix):])

            vid["frame_list"] = sorted(files, key=key)
            vid["file_prefix"] = prefix
            vid["file_ext"]   = ext
            vid["modality"]   = modality   # <— parent prepare_data keeps this

        return data_list
