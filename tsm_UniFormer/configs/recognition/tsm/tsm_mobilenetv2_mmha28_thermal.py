_base_ = [
    '../../_base_/models/tsm_mobilenet_v2.py',
    '../../_base_/default_runtime.py'
]

# MMHA-28: 28 action classes
num_classes = 28

dataset_type = 'RawframeDataset'
data_root = 'dataset/train'
data_root_val = 'dataset/val'
ann_file_train = 'dataset/train_thermal.txt'
ann_file_val = 'dataset/val_thermal.txt'
ann_file_test = 'dataset/test_thermal.txt'

file_client_args = dict(io_backend='disk')

# Thermal frame naming template
filename_tmpl = 'frame_{:04d}.jpg'

custom_imports = dict(
    imports=[
        'mmaction.datasets.custom_rawframe_dataset',
        'mmaction.datasets.transforms.custom_loading',
    ],
    allow_failed_imports=False
)

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='CustomRawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='CustomRawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='CustomRawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CustomRawframeDataset',
        modality='Thermal',
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline,
        filename_tmpl=filename_tmpl,
        file_prefix='frame_',
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CustomRawframeDataset',
        modality='Thermal',
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=val_pipeline,
        test_mode=True,
        filename_tmpl=filename_tmpl,
        file_prefix='frame_',
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CustomRawframeDataset',
        modality='Thermal',
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_val),
        pipeline=test_pipeline,
        test_mode=True,
        filename_tmpl=filename_tmpl,
        file_prefix='frame_',
    )
)

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(interval=3, max_keep_ckpts=3))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=10, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=10,
        by_epoch=True,
        milestones=[4, 8],
        gamma=0.1)
]

optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.00002),
    clip_grad=dict(max_norm=20, norm_type=2)
)

auto_scale_lr = dict(enable=True, base_batch_size=128)
