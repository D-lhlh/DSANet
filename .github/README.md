# DSANet

<div>
  <img src="DSANet.png" alt="竖图描述" width="300" />
</div>


## Performance
CItyscapes val sets:

| Methods | Size     | Params(M) | Flops(G) | Speed(FPS) | mIoU | logs | checkpoint |
|---------|----------|-----------|----------|------------|------|------|------------|
| DSANet  | 512x1024 | 1.17      | 5.39     | 110        | 75.6 |      |            |
| DSANet  | 768x1536 | 1.17      | 12.13    | 91         | 75.8 |      |            |
  

CamVid test sets:

| Methods | Size    | Params(M) | Flops(G) | Speed(FPS) | mIoU | logs | checkpoint |
|---------|---------|-----------|----------|------------|------|------|------------|
| DSANet  | 576x768 | 1.17      | 4.54     | 116        | 80.2 |      |            |
| DSANet  | 720x960 | 1.17      | 7.10     | 114        | 80.3 |      |            |


BDD100K val sets:

| Methods            | Size     | Params(M) | Flops(G) | Speed(FPS) | mIoU | logs | checkpoint |
|--------------------|----------|-----------|----------|------------|------|------|------------|
| DSANet             | 512x1024 | 1.17      | 5.39     | 110        | 52.4 |      |            |
| DSANet             | 720x1280 | 1.17      | 9.48     | 105        | 54.7 |      |            |
| DSANet(Cityscapes) | 512x1024 | 1.17      | 5.39     | 110        | 58.3 |      |            |
## Envs

通过pip安装mim
```shell
pip install openmim
```
使用mim安装mmcv2.1.0版本
```shell
mim install mmcv==2.1.0
```
随后安装mmdet，mmengine
```shell
mim install mmdet==3.3.0
mim install mmengine==0.10.7
```
最后cd到项目路径，将本项目作为mmseg包安装
```shell
cd .../DSANet
pip install -e .
```
## train:
打开tools/train.py,从目录中的configs/dsanet/下选择配置文件并运行
or
```shell
mim python tools/train.py configs/dsanet/dsanet_cityscapes_512x1024.py
```

## test：
打开tools/test.py,从目录中的configs/dsanet/下选择配置文件,再选择对应的权重并运行
or
```shell
mim python tools/test.py configs/dsanet/dsanet_cityscapes_512x1024.py ckps/checkpoints.pth
```
## checkpoints


[百度网盘](https://pan.baidu.com/s/1RWRFqprvdJ8x5-7gSGoT4g)提取码: he85\
[BaiduDisk](https://pan.baidu.com/s/1RWRFqprvdJ8x5-7gSGoT4g)password: he85

[谷歌](https://drive.google.com/drive/folders/15EwOLltTol5fEmayS8EQSHZM9hsT5ltD?usp=drive_link)\
[google](https://drive.google.com/drive/folders/15EwOLltTol5fEmayS8EQSHZM9hsT5ltD?usp=drive_link)

