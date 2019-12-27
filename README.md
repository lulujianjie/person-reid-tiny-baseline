# A Tiny Person ReID Baseline
Paper: "Bag of Tricks and A Strong Baseline for Deep Person Re-identification"[[pdf]](https://arxiv.org/abs/1903.07071)

This project refers the official code [link](https://github.com/michuanhaohao/reid-strong-baseline) and can reproduce the results as good as it on Market1501 when the input size is set to 256x128. If you find this project useful, please cite the offical paper.

```
@inproceedings{luo2019bag,
  title={Bag of Tricks and A Strong Baseline for Deep Person Re-identification},
  author={Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2019}
}
```

## Pipeline
<div align=center>
<img src='imgs/pipeline.jpg' width='800'>
</div>

## Pretrained Model
The pretrained (128x64) [model](https://pan.baidu.com/s/1FrEOT3h7lAePddFHNWIEjg) can be downloaded now.
Extraction code is **u3q5**.

## Get Started
1. `cd` to folder where you want to download this repo

2. Run `git clone https://github.com/lulujianjie/person-reid-tiny-baseline.git`

3. Install dependencies:
    - [pytorch>=0.4](https://pytorch.org/)
    - torchvision
    - cv2 (optional)


## Train

```bash
python train.py
```

## Test

```bash
python test.py
```

To get visualized reID results, first create `results` folder in log dir, then:
```bash
python get_vis_result.py

```
You will get the ranked results (query|rank1|rank2|...), like:
<div align=center>
<img src='imgs/results.png' width='600'>
</div>
