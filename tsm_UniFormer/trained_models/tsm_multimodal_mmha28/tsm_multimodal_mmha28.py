ann_file_test = '../dataset/test_all_modalities_cnn.txt'
ann_file_train = '../dataset/train_all_modalities_cnn.txt'
ann_file_val = '../dataset/val_all_modalities_cnn.txt'
auto_scale_lr = dict(base_batch_size=128, enable=True)
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmaction.datasets.custom_multimodal_dataset',
        'mmaction.datasets.transforms.custom_loading',
        'mmaction.datasets.transforms.normalize_per_modality',
        'mmaction.datasets.transforms.skip_if_event',
        'mmaction.evaluation.per_modality_acc',
        'mmaction.datasets.transforms.pack_with_modality',
    ])
data_root = '../dataset'
dataset_type = 'CustomMultimodalDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=3, max_keep_ckpts=3, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'none'
load_from = 'work_dirs/tsm_multimodal_mmha28/best_acc_top1_epoch_95.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        is_shift=True,
        num_segments=8,
        pretrained='mmcls://mobilenet_v2',
        shift_div=8,
        type='MobileNetV2TSM'),
    cls_head=dict(
        average_clips='prob',
        consensus=dict(dim=1, type='AvgConsensus'),
        dropout_ratio=0.5,
        in_channels=1280,
        init_std=0.001,
        is_shift=True,
        num_classes=28,
        num_segments=8,
        spatial_type='avg',
        type='TSMHead'),
    data_preprocessor=dict(
        format_shape='NCHW',
        mean=None,
        std=None,
        to_float32=True,
        to_rgb=False,
        type='ActionDataPreprocessor'),
    test_cfg=None,
    train_cfg=None,
    type='Recognizer2D')
num_classes = 28
optim_wrapper = dict(
    clip_grad=dict(max_norm=20, norm_type=2),
    constructor='TSMOptimWrapperConstructor',
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=2e-05),
    paramwise_cfg=dict(fc_lr5=True))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=10,
        gamma=0.1,
        milestones=[
            4,
            8,
        ],
        type='MultiStepLR'),
]
preprocess_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ], std=[
        58.395,
        57.12,
        57.375,
    ])
resume = False
seed = 42
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='../dataset/test_all_modalities_cnn.txt',
        data_prefix=dict(img='../dataset/test'),
        pipeline=[
            dict(
                clip_len=1,
                frame_interval=4,
                num_clips=8,
                test_mode=True,
                type='SampleFrames'),
            dict(io_backend='disk', type='CustomRawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=256, type='ThreeCrop'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='NormalizePerModality'),
            dict(type='PackWithModality'),
        ],
        test_mode=True,
        type='CustomMultimodalDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='AccMetric'),
    dict(type='PerModalityAccuracy'),
]
test_pipeline = [
    dict(
        clip_len=1,
        frame_interval=4,
        num_clips=8,
        test_mode=True,
        type='SampleFrames'),
    dict(io_backend='disk', type='CustomRawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=256, type='ThreeCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='NormalizePerModality'),
    dict(type='PackWithModality'),
]
train_cfg = dict(
    max_epochs=100, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='../dataset/train_all_modalities_cnn.txt',
        data_prefix=dict(img='../dataset/train'),
        pipeline=[
            dict(
                clip_len=1, frame_interval=4, num_clips=8,
                type='SampleFrames'),
            dict(io_backend='disk', type='CustomRawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(
                op=dict(
                    input_size=224,
                    max_wh_scale_gap=1,
                    num_fixed_crops=13,
                    random_crop=False,
                    scales=(
                        1,
                        0.875,
                        0.75,
                        0.66,
                    ),
                    type='MultiScaleCrop'),
                type='SkipIfEvent'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(op=dict(flip_ratio=0.5, type='Flip'), type='SkipIfEvent'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='NormalizePerModality'),
            dict(type='PackWithModality'),
        ],
        type='CustomMultimodalDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(clip_len=1, frame_interval=4, num_clips=8, type='SampleFrames'),
    dict(io_backend='disk', type='CustomRawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(
        op=dict(
            input_size=224,
            max_wh_scale_gap=1,
            num_fixed_crops=13,
            random_crop=False,
            scales=(
                1,
                0.875,
                0.75,
                0.66,
            ),
            type='MultiScaleCrop'),
        type='SkipIfEvent'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(op=dict(flip_ratio=0.5, type='Flip'), type='SkipIfEvent'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='NormalizePerModality'),
    dict(type='PackWithModality'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='../dataset/val_all_modalities_cnn.txt',
        data_prefix=dict(img='../dataset/val'),
        pipeline=[
            dict(
                clip_len=1,
                frame_interval=4,
                num_clips=8,
                test_mode=True,
                type='SampleFrames'),
            dict(io_backend='disk', type='CustomRawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='NormalizePerModality'),
            dict(type='PackWithModality'),
        ],
        test_mode=True,
        type='CustomMultimodalDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='AccMetric'),
    dict(type='PerModalityAccuracy'),
]
val_pipeline = [
    dict(
        clip_len=1,
        frame_interval=4,
        num_clips=8,
        test_mode=True,
        type='SampleFrames'),
    dict(io_backend='disk', type='CustomRawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='NormalizePerModality'),
    dict(type='PackWithModality'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/tsm_multimodal_mmha28'
