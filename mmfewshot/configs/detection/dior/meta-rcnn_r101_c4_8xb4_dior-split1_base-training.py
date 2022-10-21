_base_ = [
    '../_base_/datasets/nway_kshot/base_dior.py',
    '../_base_/schedules/schedule.py', '../meta_rcnn/meta-rcnn_r101_c4.py',
    '../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        save_dataset=False,
        dataset=dict(classes='BASE_CLASSES_SPLIT1'),
        support_dataset=dict(classes='BASE_CLASSES_SPLIT1')),
    val=dict(classes='BASE_CLASSES_SPLIT1'),
    test=dict(classes='BASE_CLASSES_SPLIT1'),
    model_init=dict(classes='BASE_CLASSES_SPLIT1'))
lr_config = dict(warmup_iters=100, step=[16000])
evaluation = dict(interval=6000)
checkpoint_config = dict(interval=6000)
runner = dict(max_iters=18000)
optimizer = dict(lr=0.00125)  # dict(lr=0.005)
# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=15, num_meta_classes=15)))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
