# 开始：安装和运行 MMSeg

## 预备知识

本教程中，我们将会演示如何使用 PyTorch 准备环境。

MMSegmentation 可以在 Linux, Windows 和 macOS 系统上运行，并且需要安装 Python 3.7+, CUDA 10.2+ 和 PyTorch 1.8+

**注意:**
如果您已经安装了 PyTorch, 可以跳过该部分，直接到[下一小节](##安装)。否则，您可以按照以下步骤操作。

**步骤 0.** 从[官方网站](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda

**步骤 1.** 创建一个 conda 环境，并激活

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** 参考 [official instructions](https://pytorch.org/get-started/locally/) 安装 PyTorch

在 GPU 平台上：
推荐torch2.0.0以获取最佳兼容性
```shell
conda install pytorch torchvision -c pytorch
```

在 CPU 平台上

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## 安装

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
## test：
打开tools/test.py,从目录中的configs/dsanet/下选择配置文件,再选择对应的权重并运行
## checkpoints
[百度网盘](https://pan.baidu.com/s/1vMV8L0v2ps4McU89bqMaOw?pwd=yq5q)提取码: yq5q

[谷歌](https://drive.google.com/file/d/1oDCCVvLzEbULaL3tNJ81dPR-LIYJj_ez/view?usp=drive_link)

