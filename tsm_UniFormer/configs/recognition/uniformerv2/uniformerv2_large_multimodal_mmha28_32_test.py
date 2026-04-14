# Test config for UniFormerV2-Large-32 on "test" dataset
_base_ = ['uniformerv2_large_multimodal_mmha28_32.py']

data_root_test = '../../cls/VideoMamba-main/videomamba/video_sm/data/new_test'
ann_file_test = '../../cls/VideoMamba-main/videomamba/video_sm/data/new_test/paths_and_labels_all.csv'

test_dataloader = dict(
    dataset=dict(
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_test),
    ),
)
