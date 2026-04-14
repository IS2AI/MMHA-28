# Test config for UniFormerV2-Large-8 on "test" dataset
_base_ = ['uniformerv2_large_multimodal_mmha28_8.py']

data_root_test = '../../cls/VideoMamba-main/videomamba/video_sm/data/test'
ann_file_test = '../../cls/VideoMamba-main/videomamba/video_sm/data/test_ALL.csv'

test_dataloader = dict(
    dataset=dict(
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_test),
    ),
)
