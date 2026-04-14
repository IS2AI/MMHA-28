import os
import os.path as osp

from mmengine.fileio import list_from_file
from mmaction.registry import DATASETS
from .rawframe_dataset import RawframeDataset


@DATASETS.register_module()
class CustomMultimodalDataset(RawframeDataset):
    """Load RGB / Depth / Thermal / Event / DepthSyn folders one by one.

    Additionally supports CSV annotation files of the form:

        data/new_test/.../depth_rgb/.../depth_images,<label>
        data/new_test/.../depth_rgb/.../depth_syn,<label>
    """

    def _build_video_info_from_dir(self, frame_dir: str, label: int) -> dict:
        """Build a single video_info entry given a frame directory."""
        if not os.path.isdir(frame_dir):
            raise FileNotFoundError(f"Directory not found: {frame_dir}")

        if "rgb_images" in frame_dir:
            prefix, modality = "rgb_", "RGB"
            exts = [".png"]
        elif "depth_images" in frame_dir:
            prefix, modality = "depth_", "Depth"
            exts = [".png"]
        elif "thermal" in frame_dir:
            prefix, modality = "frame_", "Thermal"
            exts = [".jpg", ".png"]
        elif "event-streams" in frame_dir:
            prefix, modality = "frame_", "Event"
            exts = [".npy"]
        elif "depth_syn" in frame_dir:
            prefix, modality = "rgb_", "DepthSyn"
            exts = [".jpg"]
        else:
            raise RuntimeError(f"Cannot infer modality for {frame_dir}")

        files = []
        used_ext = None
        for ext in exts:
            cand = [
                fn for fn in os.listdir(frame_dir)
                if fn.startswith(prefix) and fn.endswith(ext)
            ]
            if cand:
                files = cand
                used_ext = ext
                break

        if not files or used_ext is None:
            exts_str = ",".join(exts)
            raise FileNotFoundError(
                f"No '{prefix}*{{{exts_str}}}' files found in {frame_dir}")

        def key(fn: str) -> int:
            stem = os.path.splitext(fn)[0]
            num = stem[len(prefix):]
            return int(num) if num.isdigit() else 0

        files = sorted(files, key=key)

        video_info = dict(
            frame_dir=frame_dir,
            label=label,
            total_frames=len(files),
            frame_list=files,
            file_prefix=prefix,
            file_ext=used_ext,
            modality=modality,
        )
        return video_info

    def load_data_list(self):
        # ------------------------------------------------------------------
        # CSV format (e.g. new_test/paths_and_labels_all.csv)
        # ------------------------------------------------------------------
        if self.ann_file.endswith('.csv'):
            data_list = []
            ann_dir = osp.dirname(self.ann_file)
            lines = list_from_file(self.ann_file)

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) != 2:
                    raise ValueError(
                        f'Invalid CSV annotation line (expected "path,label"): '
                        f'{line}')

                rel_path, label_str = parts[0].strip(), parts[1].strip()

                if 'new_test/' in rel_path:
                    rel_after = rel_path.split('new_test/', 1)[1]
                    frame_dir = osp.join(ann_dir, rel_after)
                elif 'new_test_wild/' in rel_path:
                    rel_after = rel_path.split('new_test_wild/', 1)[1]
                    frame_dir = osp.join(ann_dir, rel_after)
                elif rel_path.startswith('data/'):
                    # Strip leading 'data/' since ann_dir is already the data folder
                    rel_after = rel_path[5:]  # Remove 'data/'
                    frame_dir = osp.join(ann_dir, rel_after)
                else:
                    frame_dir = osp.join(ann_dir, rel_path)

                label = int(label_str)
                try:
                    video_info = self._build_video_info_from_dir(
                        frame_dir=frame_dir, label=label)
                except FileNotFoundError:
                    continue
                data_list.append(video_info)

            return data_list

        # ------------------------------------------------------------------
        # Default TXT format via RawframeDataset
        # ------------------------------------------------------------------
        data_list = super().load_data_list()

        for vid in data_list:
            fdir = vid["frame_dir"]
            video_info = self._build_video_info_from_dir(
                frame_dir=fdir, label=vid["label"])
            vid.update(video_info)

        return data_list

# import os
# import os.path as osp

# from mmengine.fileio import list_from_file
# from mmaction.registry import DATASETS
# from .rawframe_dataset import RawframeDataset


# @DATASETS.register_module()
# class CustomMultimodalDataset(RawframeDataset):
#     """Load RGB / Depth / Thermal / Event folders one by one.

#     Additionally supports CSV annotation files of the form:

#         data/new_test/.../depth_rgb/.../depth_images,<label>

#     where we currently **skip** any rows containing ``depth_syn``.
#     """

#     def _build_video_info_from_dir(self, frame_dir: str, label: int) -> dict:
#         """Build a single video_info entry given a frame directory."""
#         if not os.path.isdir(frame_dir):
#             raise FileNotFoundError(f"Directory not found: {frame_dir}")

#         if "rgb_images" in frame_dir:
#             prefix, modality = "rgb_", "RGB"
#             exts = [".png"]
#         elif "depth_images" in frame_dir:
#             prefix, modality = "depth_", "Depth"
#             exts = [".png"]
#         elif "thermal" in frame_dir:
#             prefix, modality = "frame_", "Thermal"
#             # Support both JPG and PNG for thermal frames
#             exts = [".jpg", ".png"]
#         elif "event-streams" in frame_dir:
#             prefix, modality = "frame_", "Event"
#             exts = [".npy"]
#         else:
#             raise RuntimeError(f"Cannot infer modality for {frame_dir}")

#         files = []
#         used_ext = None
#         for ext in exts:
#             cand = [
#                 fn for fn in os.listdir(frame_dir)
#                 if fn.startswith(prefix) and fn.endswith(ext)
#             ]
#             if cand:
#                 files = cand
#                 used_ext = ext
#                 break

#         if not files or used_ext is None:
#             exts_str = ",".join(exts)
#             raise FileNotFoundError(
#                 f"No '{prefix}*{{{exts_str}}}' files found in {frame_dir}")

#         def key(fn: str) -> int:
#             stem = os.path.splitext(fn)[0]
#             num = stem[len(prefix):]
#             return int(num) if num.isdigit() else 0

#         files = sorted(files, key=key)

#         video_info = dict(
#             frame_dir=frame_dir,
#             label=label,
#             total_frames=len(files),
#             frame_list=files,
#             file_prefix=prefix,
#             file_ext=used_ext,
#             modality=modality,
#         )
#         return video_info

#     def load_data_list(self):
#         # ------------------------------------------------------------------
#         # CSV format (e.g. new_test/paths_and_labels_all.csv)
#         # ------------------------------------------------------------------
#         if self.ann_file.endswith('.csv'):
#             data_list = []
#             ann_dir = osp.dirname(self.ann_file)
#             lines = list_from_file(self.ann_file)

#             for line in lines:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 # Expect "path,label"
#                 parts = line.split(',')
#                 if len(parts) != 2:
#                     raise ValueError(
#                         f'Invalid CSV annotation line (expected "path,label"): '
#                         f'{line}')

#                 rel_path, label_str = parts[0].strip(), parts[1].strip()

#                 # Skip depth_syn modality for now
#                 if 'depth_syn' in rel_path:
#                     continue

#                 # CSV stores paths like "data/new_test/…".
#                 # We resolve them relative to the directory of the CSV file
#                 # so this stays robust to different roots.
#                 if 'new_test/' in rel_path:
#                     rel_after = rel_path.split('new_test/', 1)[1]
#                     frame_dir = osp.join(ann_dir, rel_after)
#                 else:
#                     frame_dir = osp.join(ann_dir, rel_path)
#                     label = int(label_str)
#                 try:
#                     video_info = self._build_video_info_from_dir(
#                         frame_dir=frame_dir, label=label)
#                 except FileNotFoundError:
#                     # Silently skip entries whose directories do not contain
#                     # any expected frame files (e.g. thermal folders that only
#                     # hold other modality sub-folders).
#                     continue
#                 data_list.append(video_info)

#             return data_list

#         # ------------------------------------------------------------------
#         # Default TXT format via RawframeDataset
#         # ------------------------------------------------------------------
#         data_list = super().load_data_list()

#         for vid in data_list:
#             fdir = vid["frame_dir"]
#             video_info = self._build_video_info_from_dir(
#                 frame_dir=fdir, label=vid["label"])

#             # Merge back into the original dict to keep any other keys
#             vid.update(video_info)

#         return data_list