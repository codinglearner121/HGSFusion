# HGSFusion 项目复现 环境配置教程

---

## 概述

HGSFusion 是一种用于 **3D 目标检测** 的雷达-相机融合网络，结合了激光雷达（LiDAR）和相机图像的优势，在自动驾驶等场景中具有广泛应用。为了更准确地复现其性能表现，本文档将指导你完成从零开始的环境搭建与依赖安装。

本教程参考了 [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) 的配置要求，并适配 HGSFusion 项目的实际需求，提供完整、可操作的环境初始化流程。

---

## 原项目环境初始化

以下步骤将帮助你尽可能按照原作者的开发环境进行配置，以确保实验结果的一致性。

### 系统要求

| 组件           | 推荐配置                     |
| ------------ | ------------------------ |
| 操作系统         | Ubuntu 20.04 LTS         |
| GPU          | 支持 CUDA 11.6 的 NVIDIA 显卡 |
| Python 版本    | Python 3.9.18            |
| CUDA Toolkit | 11.6                     |
| cuDNN        | 8.x                      |

> 📌 注意：CUDA Toolkit 安装请参考 [NVIDIA 官方文档](https://developer.nvidia.com/cuda-11-6-0-download-archive)：
>
> - Linux 用户请参考 [Ubuntu 20.04 安装指南](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)
> - WSL 用户请参考 [WSL-Ubuntu 安装指南](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

安装完成后，请确认 CUDA 已加入系统环境变量路径中。

---

### 快速安装流程

> ⚠️ 说明：该方式为简化版命令流，适用于熟悉 Conda/PyTorch 的用户。未经过全面测试，建议优先使用完整安装流程。

#### 步骤一：克隆仓库

```bash
git clone https://github.com/codinglearner121/HGSFusion.git

cd HGSFusion
```

#### 步骤二：创建 Conda 虚拟环境并安装依赖

##### 方法一：通过 `environment_cu116.yml` 创建环境

```bash
conda env create -f environment_cu116.yml
```

##### 方法二：手动创建并安装依赖

```bash
conda create -n hgsfusion python=3.9.18 -y && conda activate hgsfusion

pip install -r requirements.txt
```

#### 步骤三：编译 CUDA 扩展模块

进入项目目录后，需编译自定义 CUDA 扩展模块：

```bash
python setup.py develop

cd pcdet/ops/pillar_ops

python setup.py develop
```

---

### 完整安装流程

#### 步骤一：创建 Conda 虚拟环境

推荐使用 `conda` 创建独立环境，避免依赖冲突：

```bash
conda create -n hgsfusion python=3.9.18 -y

conda activate hgsfusion
```

#### 步骤二：安装 PyTorch 和 torchvision

根据你的 CUDA 版本（11.6）安装对应的 PyTorch 和 torchvision：

```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

> ✅ 可访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取其他版本支持信息。

#### 步骤三：安装项目依赖库

安装项目所需的额外 Python 包：

```bash
pip install numba openmim mmcv==2.1.0 mmdet==3.3.0 ninja scikit-image av2 pyquaternion
```

- `mmcv` 和 `mmdet` 是 OpenMMLab 生态的核心组件。

- `av2` 是 Argoverse v2 数据集的处理工具。

- `pyquaternion` 用于姿态变换计算。

#### 步骤四：克隆项目代码

```bash
git clone https://github.com/codinglearner121/HGSFusion.git

cd HGSFusion
```

#### 步骤五：编译 CUDA 扩展模块

进入项目目录后，需编译自定义 CUDA 扩展模块：

```bash
python setup.py develop

cd pcdet/ops/pillar_ops

python setup.py develop
```

这一步会构建点云处理相关的底层加速模块，确保训练/推理效率。

---

## 基于最新工具链的 HGSFusion 环境配置

本章介绍基于 **最新工具链与依赖库版本** 的 HGSFusion 项目环境配置方案，适用于希望适配前沿技术栈或解决旧版本兼容性问题的开发者。  

---

### 系统要求

| 组件           | 推荐配置                     |
| ------------ | ------------------------ |
| 操作系统         | Ubuntu 22.04 LTS 或更高     |
| GPU          | 支持 CUDA 12.8 的 NVIDIA 显卡 |
| Python 版本    | Python 3.10 或更高          |
| CUDA Toolkit | 12.8                     |
| PyTorch      | ≥2.6.0+cu128             |
| cuDNN        | 9.x                      |

> 📌 注意事项：  

> - 升级 Ubuntu 系统可参考 `do-release-upgrade` 命令。

> - CUDA Toolkit 12.8 安装指南详见 [NVIDIA 官方文档](https://developer.nvidia.com/cuda-12-8-0-download-archive) 。  

---

### 快速安装流程

#### 步骤一：克隆仓库

```bash
git clone https://github.com/codinglearner121/HGSFusion.git  

cd HGSFusion  
```

#### 步骤二：通过 `environment_cu128.yml` 创建环境

> ⚠️ **需显式指定 `conda-forge` 渠道**，以确保最新版本库的兼容性。  

```bash
conda env create -f environment_cu128.yml --channel conda-forge  
```

#### 步骤三：编译 CUDA 扩展模块

```bash
python setup.py develop  

cd pcdet/ops/pillar_ops  

python setup.py develop
```

---

### 完整安装流程

#### 步骤一：创建 Conda 虚拟环境

```bash
conda update conda

conda create -n hgsfusion -y  

conda activate hgsfusion
```

#### 步骤二：安装 PyTorch、torchvision 及项目依赖库

根据 CUDA 12.8 安装对应的 PyTorch 2.6.0：  

```bash
conda install torch torchvision openmim mmdet mmcv -c conda-forge
```

- 可通过 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取最新镜像源 。

- `openmim` 用于管理 OpenMMLab 项目依赖 。  

- `av2` 需确保支持 Argoverse v2 数据集的最新接口 。

安装完成后建议验证 PyTorch 是否支持 CUDA：  

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

若输出 `True` 则表示 PyTorch 已正确绑定 CUDA Toolkit 12.8 。  

#### 步骤三：编译 CUDA 扩展模块

```bash
python setup.py develop  

cd pcdet/ops/pillar_ops  

python setup.py develop
```

> 📌 若编译失败，请检查 CUDA Toolkit 与 PyTorch 版本是否严格匹配 。  

---

## 常见问题与解决方案（FAQ）

### Q1: CUDA 版本不匹配导致 PyTorch 安装失败？

- 请检查 `nvcc --version` 和 `nvidia-smi` 输出的 CUDA 版本是否一致。

- 若不一致，可尝试重新安装驱动或使用兼容版本的 PyTorch。

### Q2: 编译 CUDA 扩展时报错？

- 确保已正确安装 `ninja` 和 `CMake`

- 检查 CUDA Toolkit 是否在 PATH 中

### Q3: 依赖库版本冲突？

- 使用 `conda` 显式指定 `conda-forge` 渠道优先解析依赖 。  

---

## 数据准备与预处理（可选）

如需进行训练或测试，还需下载并配置对应数据集（如 KITTI、Argoverse 或 Waymo），具体结构及预处理脚本请参考 [OpenPCDet 的 GETTING_STARTED.md](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md)。

部分数据集支持如下：

- **KITTI**：经典 3D 检测数据集，适合入门与调试。

- **Waymo Open Dataset**：大规模自动驾驶数据集，包含多模态传感器信息。

- **Argoverse v2 (AV2)**：支持 BEV、点云、图像等多种输入形式。

---

📌 **参考资料**

- [OpenPCDet GitHub 仓库](https://github.com/open-mmlab/OpenPCDet)

- [NVIDIA CUDA 11.6 下载页面](https://developer.nvidia.com/cuda-11-6-0-download-archive)

- [PyTorch WHL 镜像源](https://download.pytorch.org/whl/cu116)

- [OpenMMLab 文档中心](https://mmcv.readthedocs.io/en/latest/)
