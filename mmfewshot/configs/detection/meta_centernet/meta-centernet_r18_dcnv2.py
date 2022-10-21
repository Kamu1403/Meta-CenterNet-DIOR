raise DeprecationWarning
_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4.py',
]
# model settings
model = dict(
    type='MetaCenterNet',
    backbone=dict(type='ResNetWithMetaConv', frozen_stages=2),
    neck=dict(
        type='CTResNetNeck',
        in_channel=512,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True),
    roi_head=dict(
        type='MetaCenterNetRoIHead',
        bbox_head=dict(
            type='CenterNetHead',
            num_classes=80,
            in_channel=64,
            feat_channel=64,
            num_meta_classes=80,
            meta_cls_in_channels=2048,
            with_meta_cls_loss=True,
            loss_meta=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
            loss_wh=dict(type='L1Loss', loss_weight=0.1),
            loss_offset=dict(type='L1Loss', loss_weight=1.0)),
        aggregation_layer=dict(
            type='AggregationLayer',
            aggregator_cfgs=[
                dict(
                    type='DepthWiseCorrelationAggregator',
                    in_channels=64,
                    with_fc=False)
            ])),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))
