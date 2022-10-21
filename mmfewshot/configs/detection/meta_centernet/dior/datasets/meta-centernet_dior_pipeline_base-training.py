_base_ = [
    '../../../_base_/datasets/nway_kshot/base_dior.py',
]

# We fixed the incorrect img_norm_cfg problem in the source code.
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
scale_size = (1333, 800)
distortion_cfg = dict(
    brightness_delta=32,
    contrast_range=(0.5, 1.5),
    saturation_range=(0.5, 1.5),
    hue_delta=18)

train_multi_pipelines = dict(
    query=[
        dict(type='LoadImageFromFile', color_type='color', to_float32=True),
        dict(type='LoadAnnotations', with_bbox=True),
        # dict(type='PhotoMetricDistortion', **distortion_cfg),
        dict(
            type='RandomCenterCropPad',
            crop_size=(512, 512),
            ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
            mean=[0, 0, 0],
            std=[1, 1, 1],
            to_rgb=True,
            test_pad_mode=None),
        dict(type='Resize', img_scale=scale_size, keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    support=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        # dict(type='PhotoMetricDistortion', **distortion_cfg),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='GenerateMask', target_size=(224, 224)),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=scale_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='RandomCenterCropPad',
                ratios=None,
                border=None,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=['logical_or', 31],
                test_pad_add_pix=1),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg', 'border'),
                keys=['img'])
        ])
]

data = dict(
    train=dict(
        dataset=dict(multi_pipelines=train_multi_pipelines),
        support_dataset=dict(multi_pipelines=train_multi_pipelines)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
    model_init=dict(pipeline=train_multi_pipelines['support'])
)
