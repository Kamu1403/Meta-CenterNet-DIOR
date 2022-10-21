# 这个新的配置文件继承自一个原始配置文件，只需要突出必要的修改部分即可
_base_ = 'mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=dict(num_classes=20)))

# 修改数据集相关设置
dataset_type = 'DIORDataset'
data_root = 'data/dior/'
data = dict(
    samples_per_gpu=2,
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


# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'