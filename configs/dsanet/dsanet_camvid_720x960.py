_base_ = [
    '../_base_/datasets/camvid.py',
    '../_base_/default_runtime.py'
]

crop_size = (720, 960)
class_weight = [
    1.779, 1.362, 4.616, 1.131, 2.906, 2.158, 7.213, 4.064, 3.227, 4.845, 4.871
]


num_classes = 11
norm_cfg = dict(type='BN', requires_grad=True)
act_cfg=dict(type='GELU')
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    size=crop_size,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='DSANet',
        in_channels=3,
        channels=(16, 32, 64, 128),
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        init_cfg=None,
    ),
    decode_head=dict(
        type='OHEM_HEAD',
        in_channels=32,
        channels=32,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        num_classes=num_classes,
        loss_detail_weight=1.0,
        in_index=1,
        loss_decode=[
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0),
        ]
    ),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=64,
            channels=32,
            num_convs=1,
            num_classes=num_classes,
            in_index=0,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)


optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
iters = 80000
warmup_iters = 2000
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        end_factor=1.0,
        begin=0,
        end=warmup_iters,
        by_epoch=False
    ),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=warmup_iters,
        end=iters,
        by_epoch=False
    )
]



train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(960, 720),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(960, 720), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(pipeline=train_pipeline)
)
test_pipeline = val_pipeline
val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(pipeline=val_pipeline)
)
test_dataloader = dict(
    dataset=dict(pipeline=val_pipeline)
)
val_iters = 2000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=val_iters)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=val_iters,
        save_best='mIoU', max_keep_ckpts=1,
        save_optimizer=False,
        file_client_args=None,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    #visualization=dict(type='SegVisualizationHook')
)
