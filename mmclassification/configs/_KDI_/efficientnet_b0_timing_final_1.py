dataset_type = 'CustomDataset'
classes = ['Day', 'Night']
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(380, 380)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(380, 380)),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        classes=['Day', 'Night'],
        type='CustomDataset',
        data_prefix='../data/mmcls_timing_whole/train',
        ann_file='../data/mmcls_timing_whole/meta/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(380, 380)),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        classes=['Day', 'Night'],
        type='CustomDataset',
        data_prefix='../data/mmcls_timing_whole/train',
        ann_file='../data/mmcls_timing_whole/meta/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(380, 380)),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        classes=['Day', 'Night'],
        type='CustomDataset',
        data_prefix='../data/mmcls_timing_whole/train',
        ann_file='../data/mmcls_timing_whole/meta/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(380, 380)),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientNet',
        arch='b0',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth',
            prefix='backbone')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=1))
optimizer = dict(type='AdamW', lr=5e-05, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=1,
    warmup_by_epoch=True,
    warmup_ratio=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=5)
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
work_dir = './work_dirs\efficientnet_b0_timing_final_1'
gpu_ids = [0]
