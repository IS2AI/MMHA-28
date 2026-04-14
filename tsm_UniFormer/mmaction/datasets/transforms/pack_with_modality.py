# mmaction/datasets/transforms/pack_with_modality.py
from mmaction.datasets.transforms.formatting import PackActionInputs
from mmaction.registry import TRANSFORMS

@TRANSFORMS.register_module()
class PackWithModality(PackActionInputs):
    """Keep results['modality'] both in the batch dict and inside the DataSample."""
    def __call__(self, results):
        # Cache the modality string before the parent class prunes keys
        modality = results.get('modality', None)

        packed = super().__call__(results)        # <-- this is a dict
        if modality is not None:
            # 1) keep it in the batch dict so your metric can read data_batch['modality']
            packed['modality'] = modality

            # 2) also tuck it inside the DataSample’s metainfo for completeness
            if 'data_sample' in packed:
                packed['data_sample'].set_metainfo({'modality': modality})
            elif 'data_samples' in packed:
                ds = packed['data_samples']
                if isinstance(ds, list):
                    for s in ds:
                        s.set_metainfo({'modality': modality})
                else:
                    ds.set_metainfo({'modality': modality})


        return packed
