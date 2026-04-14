# configs/uniformerv2_large_multimodal_mmha28_32.py
#
# UniFormerV2-Large (CLIP-pretrained ViT-L/14 backbone) for the MMHA-28 dataset
# with 32 frames input.

_base_ = ['../../_base_/default_runtime.py']

import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ---------------------------------------------------------------------- #
#  Custom Python modules (all live under mmaction/)
# ---------------------------------------------------------------------- #
custom_imports = dict(
    imports=[
        'mmaction.datasets.custom_multimodal_dataset',
        'mmaction.datasets.transforms.custom_loading',
        'mmaction.datasets.transforms.normalize_per_modality',
        'mmaction.datasets.transforms.skip_if_event',
        'mmaction.evaluation.per_modality_acc',
        'mmaction.datasets.transforms.pack_with_modality',
    ],
    allow_failed_imports=False,
)

# ---------------------------------------------------------------------- #
#  Model settings (UniFormerV2-Large, 32 frames)
# ---------------------------------------------------------------------- #
num_frames = 32
num_classes = 28

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='UniFormerV2',
        input_resolution=224,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        t_size=num_frames,
        dw_reduction=1.5,
        backbone_drop_path_rate=0.,
        temporal_downsample=False,
        no_lmhra=True,
        double_lmhra=True,
        return_list=[20, 21, 22, 23],
        n_layers=4,
        n_dim=1024,
        n_head=16,
        mlp_factor=4.,
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5],
        clip_pretrained=True,
        pretrained='ViT-L/14',
    ),
    cls_head=dict(
        type='UniFormerHead',
        dropout_ratio=0.5,
        num_classes=num_classes,
        in_channels=1024,
        average_clips='prob',
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=None,
        std=None,
        to_rgb=False,
        to_float32=True,
        format_shape='NCTHW',
    ),
)

# ---------------------------------------------------------------------- #
#  Basic dataset / task info
# ---------------------------------------------------------------------- #
dataset_type = 'CustomMultimodalDataset'
data_root = '../../cls/VideoMamba-main/videomamba/video_sm/data'
data_root_test = '../../cls/VideoMamba-main/videomamba/video_sm/data/new_test'
ann_file_train = f'../../cls/dataset/train_all_modalities_cnn.txt'
ann_file_val   = f'../../cls/dataset/val_all_modalities_cnn.txt'
ann_file_test  = f'{data_root_test}/paths_and_labels_all.csv'

file_client_args = dict(io_backend='disk')

# ---------------------------------------------------------------------- #
#  Pipelines
# ---------------------------------------------------------------------- #
train_pipeline = [
    dict(type='SampleFrames', clip_len=num_frames, frame_interval=4, num_clips=1),
    dict(type='CustomRawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='SkipIfEvent',
        op=dict(
            type='MultiScaleCrop',
            input_size=224,
            scales=(1, 0.875, 0.75, 0.66),
            random_crop=False,
            max_wh_scale_gap=1,
            num_fixed_crops=13,
        ),
    ),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='SkipIfEvent', op=dict(type='Flip', flip_ratio=0.5)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='NormalizePerModality'),
    dict(type='PackWithModality'),
]

val_pipeline = [
    dict(type='SampleFrames', clip_len=num_frames, frame_interval=4, num_clips=1, test_mode=True),
    dict(type='CustomRawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='NormalizePerModality'),
    dict(type='PackWithModality'),
]

test_pipeline = [
    dict(type='SampleFrames', clip_len=num_frames, frame_interval=4, num_clips=1, test_mode=True),
    dict(type='CustomRawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='NormalizePerModality'),
    dict(type='PackWithModality'),
]

# ---------------------------------------------------------------------- #
#  DataLoaders (reduced batch size for Large model + 32 frames)
# ---------------------------------------------------------------------- #
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=f'{data_root}/train'),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=f'{data_root}/val'),
        pipeline=val_pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_test),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

# ---------------------------------------------------------------------- #
#  Evaluation
# ---------------------------------------------------------------------- #
val_evaluator = [dict(type='AccMetric'), dict(type='PerModalityAccuracy')]
test_evaluator = val_evaluator

# ---------------------------------------------------------------------- #
#  Optimisation / LR schedule
# ---------------------------------------------------------------------- #
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        eta_min_ratio=0.1,
        by_epoch=True,
        begin=5,
        end=100,
        convert_to_iter_based=True,
    ),
]

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=20, norm_type=2),
)

auto_scale_lr = dict(enable=False)

# ---------------------------------------------------------------------- #
#  Runtime hooks & loops
# ---------------------------------------------------------------------- #
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3),
    logger=dict(interval=100),
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

