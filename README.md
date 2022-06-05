# XBound-Former: Toward Cross-scale Boundary Modeling in Transformers

## Introduction

This is an official release of the paper **XBound-Former: Toward Cross-scale Boundary Modeling in Transformers**, including the network implementation and the training scripts.

<div align="center" border=> <img src=frame.jpg width="700" > </div>

## News
- **[5/27 2022] We have released the training scripts.**
- **[5/19 2022] We have created this repo.**

## Code List

- [x] Network
- [x] Pre-processing
- [x] Training Codes
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

You need to change the **File Path** to your own and select the correct function.

### Training 

Please run:

```bash
$ python src/train.py
```
You need to change the **File Path** to your own and select the correct function.

### Testing

Download the pretrained weight for ISCI-2016&$ph^2$ dataset from [Google Drive](https://drive.google.com/file/d/1-eMHYX1fr-QvI3n50S0xqWcxc3FGsMgE/view?usp=sharing) and move to the logger dir.

Then, please run:

```bash
$ python src/test.py
```

### Result
The ISIC-2016&$ph^2$ dataset:
<div align="center" border=> <img src=isic2016.png width="700" > </div>

## Citation

If you find XBound-Former useful in your research, please consider citing:
```
@ARTICLE{2022arXiv220600806W,
       author = {{Wang}, Jiacheng and {Chen}, Fei and {Ma}, Yuxi and {Wang}, Liansheng and {Fei}, Zhaodong and {Shuai}, Jianwei and {Tang}, Xiangdong and {Zhou}, Qichao and {Qin}, Jing},
        title = "{XBound-Former: Toward Cross-scale Boundary Modeling in Transformers}",
      journal = {arXiv e-prints},
         year = 2022,
        month = jun,
          eid = {arXiv:2206.00806},
        pages = {arXiv:2206.00806},
archivePrefix = {arXiv},
       eprint = {2206.00806},
 primaryClass = {cs.CV},
}


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
