ann_file_test = '../../cls/VideoMamba-main/videomamba/video_sm/data/new_test/paths_and_labels_all.csv'
ann_file_train = '../../cls/dataset/train_all_modalities_cnn.txt'
ann_file_val = '../../cls/dataset/val_all_modalities_cnn.txt'
auto_scale_lr = dict(enable=False)
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
data_root = '../../cls/dataset'
data_root_test = '../../cls/VideoMamba-main/videomamba/video_sm/data/new_test'
dataset_type = 'CustomMultimodalDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=5, max_keep_ckpts=3, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=100, type='LoggerHook'),
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
load_from = 'work_dirs/uniformerv2_large_multimodal_mmha28_16/best_acc_top1_epoch_90.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        backbone_drop_path_rate=0.0,
        clip_pretrained=True,
        double_lmhra=True,
        drop_path_rate=0.0,
        dw_reduction=1.5,
        heads=16,
        input_resolution=224,
        layers=24,
        mlp_dropout=[
            0.5,
            0.5,
            0.5,
            0.5,
        ],
        mlp_factor=4.0,
        n_dim=1024,
        n_head=16,
        n_layers=4,
        no_lmhra=True,
        patch_size=14,
        pretrained='ViT-L/14',
        return_list=[
            20,
            21,
            22,
            23,
        ],
        t_size=16,
        temporal_downsample=False,
        type='UniFormerV2',
        width=1024),
    cls_head=dict(
        average_clips='prob',
        dropout_ratio=0.5,
        in_channels=1024,
        num_classes=28,
        type='UniFormerHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=None,
        std=None,
        to_float32=True,
        to_rgb=False,
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
num_classes = 28
num_frames = 16
optim_wrapper = dict(
    clip_grad=dict(max_norm=20, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=1e-05, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.1,
        type='LinearLR'),
    dict(
        T_max=95,
        begin=5,
        by_epoch=True,
        convert_to_iter_based=True,
        end=100,
        eta_min_ratio=0.1,
        type='CosineAnnealingLR'),
]
resume = False
seed = 42
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '../../cls/VideoMamba-main/videomamba/video_sm/data/new_test/paths_and_labels_all.csv',
        data_prefix=dict(
            img='../../cls/VideoMamba-main/videomamba/video_sm/data/new_test'),
        pipeline=[
            dict(
                clip_len=16,
                frame_interval=4,
                num_clips=1,
                test_mode=True,
                type='SampleFrames'),
            dict(io_backend='disk', type='CustomRawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='NormalizePerModality'),
            dict(type='PackWithModality'),
        ],
        test_mode=True,
        type='CustomMultimodalDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='AccMetric'),
    dict(type='PerModalityAccuracy'),
]
test_pipeline = [
    dict(
        clip_len=16,
        frame_interval=4,
        num_clips=1,
        test_mode=True,
        type='SampleFrames'),
    dict(io_backend='disk', type='CustomRawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='NormalizePerModality'),
    dict(type='PackWithModality'),
]
train_cfg = dict(
    max_epochs=100, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='../../cls/dataset/train_all_modalities_cnn.txt',
        data_prefix=dict(img='../../cls/dataset/train'),
        pipeline=[
            dict(
                clip_len=16,
                frame_interval=4,
                num_clips=1,
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
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='NormalizePerModality'),
            dict(type='PackWithModality'),
        ],
        type='CustomMultimodalDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(clip_len=16, frame_interval=4, num_clips=1, type='SampleFrames'),
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
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='NormalizePerModality'),
    dict(type='PackWithModality'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='../../cls/dataset/val_all_modalities_cnn.txt',
        data_prefix=dict(img='../../cls/dataset/val'),
        pipeline=[
            dict(
                clip_len=16,
                frame_interval=4,
                num_clips=1,
                test_mode=True,
                type='SampleFrames'),
            dict(io_backend='disk', type='CustomRawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='NormalizePerModality'),
            dict(type='PackWithModality'),
        ],
        test_mode=True,
        type='CustomMultimodalDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='AccMetric'),
    dict(type='PerModalityAccuracy'),
]
val_pipeline = [
    dict(
        clip_len=16,
        frame_interval=4,
        num_clips=1,
        test_mode=True,
        type='SampleFrames'),
    dict(io_backend='disk', type='CustomRawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
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
work_dir = './work_dirs/uniformerv2_large_multimodal_mmha28_16'
