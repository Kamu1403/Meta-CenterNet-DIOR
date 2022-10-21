_base_ = [
    './datasets/meta-centernet_dior_pipeline_fine-tuning.py',
    '../../_base_/schedules/schedule.py', '../meta-centernet_r50_dcnv2.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        save_dataset=True,
        dataset=dict(
            type='DIORFewShotDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='SPLIT1_5SHOT')],
            num_novel_shots=5,
            num_base_shots=5,
            classes='ALL_CLASSES_SPLIT1',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'),
    model_init=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=500, class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=500)
optimizer = dict(lr=0.0005)  # dict(lr=0.001)
lr_config = dict(warmup=None, step=[5000])
runner = dict(max_iters=5000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/meta-centernet_r50_c4_8xb4_dior-split1_base-training/latest.pth'
# model settings
# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=20, num_meta_classes=20)),
    frozen_parameters=[
        'backbone', 'neck', 'support_neck', 'aggregation_layer'
    ])

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
