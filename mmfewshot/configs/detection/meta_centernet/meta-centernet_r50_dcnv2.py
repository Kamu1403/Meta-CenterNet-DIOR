# model settings
norm_cfg = dict(type='BN', requires_grad=False)
pretrained = 'open-mmlab://detectron2/resnet50_caffe'
model = dict(
    type='MetaCenterNet',
    pretrained=pretrained,
    backbone=dict(
        type='ResNetWithMetaConv',
        frozen_stages=2,
        depth=50,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2,),
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='caffe',
    ),
    neck=dict(
        type='CTResNetNeck',
        in_channel=1024,
        num_deconv_filters=(512, 256, 64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True),
    support_neck=dict(
        type='MetaCenterNetSupportNeck',
        out_channels=2048,
        pretrained=pretrained,
        depth=50,
        stage=3,
        stride=2,
        dilation=1,
        style='caffe',
        norm_cfg=norm_cfg,
        norm_eval=True),
    roi_head=dict(
        type='MetaCenterNetRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=0,
            featmap_strides=[16]),
        bbox_head=dict(
            type='MetaCenterNetHead',
            num_classes=80,
            in_channel=96,
            feat_channel=64,
            loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
            loss_wh=dict(type='L1Loss', loss_weight=0.1),
            loss_offset=dict(type='L1Loss', loss_weight=1.0),

            with_meta_cls_loss=True,
            num_meta_classes=80,
            meta_cls_in_channels=2048,
            loss_meta=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ),
        aggregation_layer=dict(
            type='AggregationLayer',
            aggregator_cfgs=[
                dict(
                    type='CorrelationAggregator',
                    in_channels=64,
                    out_channels=96,
                    with_fc=True)
            ])),
    train_cfg=None,
    test_cfg=dict(rcnn=dict(topk=100, local_maximum_kernel=3, max_per_img=100)))
