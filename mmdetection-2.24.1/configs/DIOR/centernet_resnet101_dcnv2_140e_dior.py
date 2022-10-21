_base_ = '../centernet/centernet_resnet18_dcnv2_140e_coco.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    neck=dict(
        in_channel=2048),
    bbox_head=dict(
        num_classes=20))

dataset_type = 'DIORDataset'
data_root = 'data/dior/'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    train=dict(
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'ImageSets/Main/train.txt',
            img_prefix=data_root),
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        img_prefix=data_root),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/test.txt',
        img_prefix=data_root))

evaluation = dict(interval=5, metric='mAP')

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

