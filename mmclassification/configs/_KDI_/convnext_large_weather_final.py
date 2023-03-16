dataset_type = 'CustomDataset'
classes = ['Normal', 'Snowy', 'Rainy']
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(512, 512)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(512, 512)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        classes=['Normal', 'Snowy', 'Rainy'],
        type='CustomDataset',
        data_prefix='../data/mmcls_weather_labelingnocrash/train',
        ann_file='../data/mmcls_weather_labelingnocrash/meta/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(512, 512)),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        classes=['Normal', 'Snowy', 'Rainy'],
        type='CustomDataset',
        data_prefix='../data/mmcls_weather_labelingnocrash/val',
        ann_file='../data/mmcls_weather_labelingnocrash/meta/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(512, 512)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        classes=['Normal', 'Snowy', 'Rainy'],
        type='CustomDataset',
        data_prefix='../data/mmcls_weather_labelingnocrash/val',
        ann_file='../data/mmcls_weather_labelingnocrash/meta/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(512, 512)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ConvNeXt',
        arch='large',
        out_indices=(3, ),
        drop_path_rate=0.5,
        gap_before_final_norm=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_in21k_20220124-41b5a79f.pth',
            prefix='backbone')),
    head=dict(
        type='LinearClsHead',
        num_classes=3,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
optimizer = dict(type='AdamW', lr=5e-06, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=1,
    warmup_by_epoch=True,
    warmup_ratio=1e-06)
runner = dict(type='EpochBasedRunner', max_epochs=20)
evaluation = dict(interval=1, metric='f1_score')
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs\convnext_large_weather_final'
gpu_ids = [0]
