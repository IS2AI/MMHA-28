# Test config for UniFormerV2-Large-8 on "new_test_wild" dataset
_base_ = ['uniformerv2_large_multimodal_mmha28_8.py']

data_root_test = '../../cls/VideoMamba-main/videomamba/video_sm/data/new_test_wild'
ann_file_test = f'{data_root_test}/paths_and_labels_all.csv'

test_dataloader = dict(
    dataset=dict(
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_test),
    ),
)
