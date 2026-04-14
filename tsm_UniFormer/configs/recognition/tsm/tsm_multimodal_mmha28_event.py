# configs/tsm_multimodal_mmha28.py
#
# Joint-feature TSM (MobileNetV2 backbone) for the MMHA-28 dataset.
# Spatial augmentations (MultiScaleCrop + Flip) are wrapped in SkipIfEvent
# so they are **disabled for Event-stream (.npy) videos** but remain active
# for RGB / Depth / Thermal.

_base_ = [
    '../../_base_/models/tsm_mobilenet_v2.py',
    '../../_base_/default_runtime.py',
]

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
#  Basic dataset / task info
# ---------------------------------------------------------------------- #
num_classes = 28
dataset_type = 'CustomMultimodalDataset'
data_root = '../dataset'
ann_file_train = f'{data_root}/train_event_only_cnn.txt'
ann_file_val   = f'{data_root}/val_event_only_cnn.txt'
ann_file_test  = f'{data_root}/test_event_only_cnn.txt'

file_client_args = dict(io_backend='disk')

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
        'mmaction.datasets.transforms.pack_with_modality'
    ],
    allow_failed_imports=False,
)

# ---------------------------------------------------------------------- #
#  Pipelines
# ---------------------------------------------------------------------- #
# --- TRAIN ---
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=4, num_clips=8),
    dict(type='CustomRawFrameDecode', io_backend='disk'),

    # resize shorter side → 256 (kept for all modalities)
    dict(type='Resize', scale=(-1, 256)),

    # Multi-scale crop *unless* the sample is Event-stream
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

    # final spatial size 224×224
    dict(type='Resize', scale=(224, 224), keep_ratio=False),

    # Random flip *unless* Event
    dict(
        type='SkipIfEvent',
        op=dict(type='Flip', flip_ratio=0.5),
    ),

    dict(type='FormatShape', input_format='NCHW'),
    dict(type='NormalizePerModality'),
    dict(type='PackWithModality'),
]

# --- VALIDATION ---
val_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=4, num_clips=8, test_mode=True),
    dict(type='CustomRawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='NormalizePerModality'),
    dict(type='PackWithModality'),
]

# --- TESTING ---
test_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=4, num_clips=8, test_mode=True),
    dict(type='CustomRawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='NormalizePerModality'),
    dict(type='PackWithModality'),
]

# ---------------------------------------------------------------------- #
#  DataLoaders
# ---------------------------------------------------------------------- #
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
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
    batch_size=16,
    num_workers=8,
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
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(img=f'{data_root}/test'),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

# ---------------------------------------------------------------------- #
#  Evaluation
# ---------------------------------------------------------------------- #
val_evaluator = [
    dict(type='AccMetric'),
    dict(type='PerModalityAccuracy')
]
test_evaluator = val_evaluator

# ---------------------------------------------------------------------- #
#  Optimisation / LR schedule
# ---------------------------------------------------------------------- #
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=10,
        by_epoch=True,
        milestones=[4, 8],
        gamma=0.1,
    ),
]

optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=2e-5),
    clip_grad=dict(max_norm=20, norm_type=2),
)

auto_scale_lr = dict(enable=True, base_batch_size=128)

# ---------------------------------------------------------------------- #
#  Runtime hooks & loops
# ---------------------------------------------------------------------- #
default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=3),
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_begin=1,
    val_interval=1,
)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')