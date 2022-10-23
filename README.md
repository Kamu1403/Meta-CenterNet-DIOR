## Introduction

- This codebase is created to build benchmarks for few-shot learning object detection on 
  [**DIOR dataset**](https://www.sciencedirect.com/science/article/pii/S0924271619302825)
  and provide a new model 
  [**Meta-CenterNet**](/mmfewshot/mmfewshot/detection/models/detectors/meta_centernet.py). 
- It is modified from mmfewshot.

## Our work

- The [loading module](/mmdetection-2.24.1/mmdet/datasets/dior.py) of **DIOR dataset** is 
  provided in mmdetection module, and the configuration trained on DIOR dataset using 
  [**CenternNet**, **Faster R-CNN**, and **Mask R-CNN**](/mmdetection-2.24.1/configs/DIOR) 
  is set up.
- The [loading module](/mmfewshot/mmfewshot/detection/datasets/dior.py) of **N-way K-shot 
  DIOR dataset** is provided in mmfewshot module, and four sample category segmentation 
  is set according to the paper.  
  In addition, the [configuration](/mmfewshot/configs/detection/dior) that uses 
  **Meta-RCNN** to train on DIOR dataset is set.
- A new model 
  [**Meta-CenterNet**](/mmfewshot/mmfewshot/detection/models/detectors/meta_centernet.py) 
  is provided in mmfewshot module. Resnet and deconvolution network are used to extract 
  feature map and supporting features.  A new feature aggregation module 
  [**CorrelationAggregator**](/mmfewshot/mmfewshot/detection/models/utils/aggregation_layer.py)
  is used to convolve and combine the two features to obtain the aggregated feature map. 
  And the detection box is extracted  from the anchor-free detection head based on CenterNet. 
  The configuration for training on DIOR dataset is set.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Dependencies
Please refer to [requirements.txt](requirements.txt) or follow the setup instruction of 
[mmfewshot](https://github.com/open-mmlab/mmfewshot)

## Installation & Dataset Preparation

### Install
MMFewShot depends on [PyTorch](https://pytorch.org/) and 
[MMCV](https://github.com/open-mmlab/mmcv).
Please refer to [install.md](/mmfewshot/docs/en/install.md) for installation of MMFewShot

### Dataset

Download DIOR dataset and place it under [dataset folder](/mmfewshot/data). It should be like
```
/mmfewshot/data/dior/
        Annnotations/
                Horizontal_Bounding_Boxes/*.xml
                Oriented_Bounding_Boxes/*.xml
        ImageSets/Main/
                test.txt  
                train.txt  
                val.txt
        JPEGImages/*.jpg
```


## Getting Started

### DIOR datasets
- To use DIOR datasets in object detection projects, the following can be run, 
  to train CenterNet with DIOR dataset
- You can train on other networks by replacing the following Settings with
  - centernet_resnet101_dcnv2_140e_dior.py
  - centernet_resnet50_dcnv2_140e_dior.py            
  - centernet_resnet18_dcnv2_140e_dior.py<br>
  - faster_rcnn_r101_caffe_fpn_1x_dior.py
  - mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_dior.py
```
cd mmdetection-2.24.1/
python tools/train.py configs/DIOR/centernet_resnet101_dcnv2_140e_dior.py
```

### Few-shot DIOR datasets
- Using DIOR dataset in fewshot learning, the following can be run, 
  to train Meta-RCNN using DIOR query suport dataset
```
cd mmfewshot/
python tools/detection/train.py configs/detection/dior/meta-rcnn_r101_c4_8xb4_dior-split1_base-training.py      #base training
python tools/detection/train.py configs/detection/dior/meta-rcnn_r101_c4_8xb4_dior-split1_5shot-fine-tuning.py  #fine-tuning
```
### Use Meta-CenterNet
- Training DIOR query suport dataset on the new implemented Meta-CenterNet can run the 
  following
```
cd mmfewshot/
python tools/detection/train.py configs/detection/meta_centernet/dior/meta-centernet_r50_c4_8xb4_dior-split1_base-training.py       #base training
python tools/detection/train.py configs/detection/meta_centernet/dior/meta-centernet_r50_c4_8xb4_dior-split1_5shot-fine-tuning.py   #fine-tuning
```

## Citing

- DIOR dataset
```
@article{li2020object,
  title={Object detection in optical remote sensing images: A survey and a new benchmark},
  author={Li, Ke and Wan, Gang and Cheng, Gong and Meng, Liqiu and Han, Junwei},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={159},
  pages={296--307},
  year={2020},
  publisher={Elsevier}
}
```


## Thanks to the Third Party Libs

[Pytorch](https://pytorch.org/)

[mmfewshot](https://github.com/open-mmlab/mmfewshot)