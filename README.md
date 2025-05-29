# FMMP (CVPR'2025)

Official code repository for the paper:  
[**Recurrent Feature Mining and Keypoint Mixup Padding for Category-Agnostic Pose Estimation**](https://arxiv.org/abs/2503.21140)  
[Junjie Chen, Weilong Chen, Yifan Zuo, Yuming Fang] 

### Abstract

Category-agnostic pose estimation aims to locate keypoints on query images according to a few annotated support images for arbitrary novel classes. Existing methods generally extract support features via heatmap pooling, and obtain interacted features from support and query via cross-attention. Hence, these works neglect to mine fine-grained and structure-aware (FGSA) features from both support and query images, which are crucial for pixel-level keypoint localization. To this end, we propose a novel yet concise framework, which recurrently mines FGSA features from both support and query images. Specifically, we design a FGSA mining module based on deformable attention mechanism. On the one hand, we mine fine-grained features by applying deformable attention head over multi-scale feature maps. On the other hand, we mine structure-aware features by offsetting the reference points of keypoints to their linked keypoints. By means of above module, we recurrently mine FGSA features from support and query images, and thus obtain better support features and query estimations. In addition, we propose to use mixup keypoints to pad various classes to a unified keypoint number, which could provide richer supervision than the zero padding used in existing works. We conduct extensive experiments and in-depth studies on large-scale MP-100 dataset, and outperform SOTA method dramatically.

## Usage

### Install
The installation is similar to [CapeFormer](https://github.com/flyinglynx/CapeFormer), detailed packages could be found in `cape_environment.yml`.

### Data preparation
Please follow the [official guide](https://github.com/luminxu/Pose-for-Everything) to prepare the MP-100 dataset for training and evaluation, and organize the data structure properly. 

Alternatively, we employ an unified annotation file (i.e., `unified_ann_file.json`) and adopt valid_class_ids to set various splits.

### Training and Test

The scripts are similar to [CapeFormer](https://github.com/flyinglynx/CapeFormer), and detailed scripts could be found in `install.sh`.
Pretrained weights (e.g., mAP: ~78.7%) are available at [GtYe](https://pan.quark.cn/s/8ab1649bcee7).

## Citation
```bibtex
@inproceedings{FMMP,
  title={Recurrent Feature Mining and Keypoint Mixup Padding for Category-Agnostic Pose Estimation},
  author={Junjie Chen, Weilong Chen, Yifan Zuo, Yuming Fang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## Acknowledgement

Thanks to:

- [MMPose](https://github.com/open-mmlab/mmpose)
- [Pose-for-Everything](https://github.com/luminxu/Pose-for-Everything)
- [CapeFormer](https://github.com/flyinglynx/CapeFormer)

## License

This project is released under the [Apache 2.0 license](LICENSE).
