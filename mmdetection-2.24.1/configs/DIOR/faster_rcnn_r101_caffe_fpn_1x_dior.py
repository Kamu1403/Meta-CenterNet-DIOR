_base_ = '../faster_rcnn/faster_rcnn_r101_caffe_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=20)))

dataset_type = 'DIORDataset'
data_root = 'data/dior/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/train.txt',
        img_prefix=data_root),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        img_prefix=data_root),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/test.txt',
        img_prefix=data_root))

evaluation = dict(interval=5, metric='mAP')
runner = dict(type='EpochBasedRunner', max_epochs=20)
