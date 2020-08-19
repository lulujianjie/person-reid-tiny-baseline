# Tiny Person ReID Baseline
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

## Updates (Difference from Official Code)
* v0.1.2 (Feb. 2020)
    - Support Harder Example Mining, which achieve better performance. See [docs](https://tiny-reid.readthedocs.io/en/latest/loss/triplet.html#harder-example-mining) for details.
    - Support visualizing augmented data
    - Support flipped features
    - Support search best parameters for reranking
* v0.1.1 (Sep. 2019)
    - Support ArcFace loss
    - Support visualizing reID results
    - Add comments in config.py
* v0.1.0 (Jun. 2019)
    - Develop based on the [pytorch template](https://github.com/lulujianjie/pytorch-project-template) 
    - No need to install ignite and yacs
    - Support computing distance using cosine similarity
    - Set hyperparameters using a configuration class
    - Only support ResNet50 as the backbone

## Directory layout

    .
    ├── config                  # hyperparameters settings
    │   └── ...                 
    ├── datasets                # data loader
    │   └── ...           
    ├── log                     # log and model weights             
    ├── loss                    # loss function code
    │   └── ...   
    ├── model                   # model
    │   └── ...  
    ├── processor               # training and testing procedures
    │   └── ...    
    ├── solver                  # optimization code
    │   └── ...   
    ├── tools                   # tools
    │   └── ...
    ├── utils                   # metrics code
    │   └── ...
    ├── train.py                # train code 
    ├── test.py                 # test code 
    ├── get_vis_result.py       # get visualized results 
    ├── docs                    # docs for readme              
    └── README.md


## Pipeline
<div align=center>
<img src='docs/pipeline.jpg' width='800'>
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
python ./tools/get_vis_result.py

```
You will get the ranked results (query|rank1|rank2|...), like:
<div align=center>
<img src='docs/results.png' width='600'>
</div>

## Results

|model|method|mAP|Rank1|
|---- |----  |----|----|
|resnet50|triplet loss + softmax + center loss (B1)| 85.8| 94.1 |
|resnet50|B1 + flipped feature| 86.3| 93.9 |
|resnet50|B1 + **Harder Example Mining**| 86.2| 94.4 |
|resnet50|B1 + flipped feature + **Harder Example Mining**| 86.6| 94.6 |
|resnet50|B1 + **Harder Example Mining** + reranking| 94.1| 95.6 |
|resnet50|B1 + **Harder Example Mining** + searched reranking| 94.2| 95.8 |
