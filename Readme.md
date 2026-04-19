# Fashion-MNIST 三层神经网络分类器 (From Scratch)
本项目为李茗瑄24300980036计算机视觉第一次作业。完全从零开始（不依赖 PyTorch/TensorFlow 等自动微分框架），仅使用 NumPy 手工搭建三层神经网络（MLP），实现前向传播、反向传播、多种激活函数（ReLU, Sigmoid, Leaky ReLU, GELU）切换及 L2 正则化，并在 Fashion-MNIST 数据集完成训练与评估。

## 1. 环境依赖 (Requirements)
运行本项目需 Python 3.x 及以下基础库。可通过 `pip install numpy matplotlib scikit-learn` 命令一键安装。

| 库名称 | 核心用途 |
| :--- | :--- |
| **numpy** | 核心矩阵运算与手工反向传播梯度推导 |
| **matplotlib** | 绘制训练过程 Loss/Accuracy 曲线及权重空间可视化 |
| **scikit-learn** | 仅用于测试阶段生成和绘制混淆矩阵 |

## 2. 项目结构 (Project Structure)
| 目录 / 文件          | 功能说明 |
|:-----------------| :--- |
| `data/`          | 存放 Fashion-MNIST 原始 `.gz` 格式数据集 |
| `weights/`       | 自动保存验证集准确率最高的模型权重 (`best_model.npz`) |
| `data_loader.py` | 数据读取与预处理模块（包含张量展平、归一化、划分验证集） |
| `model.py`       | 核心模型结构（定义 MLP 类，含前向传播、损失计算与反向传播推导） |
| `train.py`       | 训练主干（包含 SGD 优化器、学习率衰减、超参数查找及学习曲线绘制） |
| `evaluate.py`    | 测试评估（导入最佳权重，输出准确率、混淆矩阵、错例与权重可视化） |

## 3. 快速运行 (How to Run)
**步骤一：超参数查找与模型训练**
执行以下命令开启训练。脚本会自动循环测试不同激活函数的表现，将最优模型权重保存到 `weights/best_model.npz` 中，并生成训练对比图。
```bash
python train.py
```
**步骤二：模型测试与结果可视化**
训练完成后执行以下命令。脚本将加载最优模型，在独立测试集上进行评估，并依次弹出“混淆矩阵”、“权重空间模式可视化”以及“错例分析”图像。
```bash
python evaluate.py
```

## 4. 预训练权重下载
最优模型权重已上传至 Google Drive，可通过 [点击此处下载](#) ，下载后将其放置于 `weights/` 文件夹下即可直接跳过训练进行测试。