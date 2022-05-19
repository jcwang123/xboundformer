# Cross-scale Boundary-aware Transformer for Skin Lesion Segmentation

## Introduction

This is an official release of the paper **Cross-scale Boundary-aware Transformer for Skin Lesion Segmentation**. 

<div align="center" border=> <img src=frame.jpg width="400" > </div>

## News

- **[5/19 2022] We have created this repo.**

## Code List

- [x] Network
- [x] Pre-processing
- [ ] Training Codes
- [ ] Pretrained Weights

For more details or any questions, please feel easy to contact us by email (jiachengw@stu.xmu.edu.cn).


## Usage

### Dataset

Please download the dataset from [ISIC](https://www.isic-archive.com/) challenge and [PH2](https://www.fc.up.pt/addi/ph2%20database.html) website.

### Pre-processing

Please run:

```bash
$ python utils/resize.py
```

You need to change the **File Path** to your own.

### Training 

### Testing

Download the pretrained weight for PH2 dataset from [Google Drive](https://drive.google.com/file/d/1-eMHYX1fr-QvI3n50S0xqWcxc3FGsMgE/view?usp=sharing).

Then, please run:

```bash
```

### Result

## Citation

If you find BAT useful in your research, please consider citing:
```
```
and the prior work, BAT, as:
```
@inproceedings{wang2021boundary,
  title={Boundary-Aware Transformers for Skin Lesion Segmentation},
  author={Wang, Jiacheng and Wei, Lan and Wang, Liansheng and Zhou, Qichao and Zhu, Lei and Qin, Jing},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={206--216},
  year={2021},
  organization={Springer}
}
```
