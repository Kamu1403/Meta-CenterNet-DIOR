_base_ = [
    '../_base_/datasets/nway_kshot/few_shot_dior.py',
    '../_base_/schedules/schedule.py', '../meta_rcnn/meta-rcnn_r101_c4.py',
    '../_base_/default_runtime.py'
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
    'work_dirs/meta-rcnn_r101_c4_8xb4_dior-split1_base-training/latest.pth'
# model settings
model = dict(frozen_parameters=[
    'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
])

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
