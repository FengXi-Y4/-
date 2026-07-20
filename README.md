# -
# 可见光消色差超透镜

本项目用于基于 CST 仿真数据和深度学习模型实现可见光消色差超透镜的逆向设计与版图生成。代码整体流程分为四步：从 CST 工程提取数据，构建训练数据集，训练相位预测模型与相位偏移模型，最后根据目标焦距和波长分布生成超透镜设计结果。

## 项目功能

项目当前包含以下核心能力：

1. 从 CST 工程文件中提取纳米柱结构参数、频率响应和相位数据。
2. 对原始相位做平移处理，生成更适合建模的训练样本。
3. 构建两类数据集：相位平移数据集和相位偏移数据集。
4. 训练两个神经网络模型：
	- `metalens_model.py`：根据相位预测结构参数 L、W。
	- `metalens_phase_shift_model.py`：根据结构参数预测相位偏移量。
5. 使用训练好的模型生成超透镜设计，并输出 L/W 分布图、相位图和导出文件。

## 代码结构

- [config.py](config.py)：统一管理数据目录、模型目录和超透镜设计参数。
- [DataPrePro_PhaseShift.py](DataPrePro_PhaseShift.py)：从 CST 工程提取数据并执行相位平移预处理。
- [CreateDataset_ShiftedPhase.py](CreateDataset_ShiftedPhase.py)：基于平移后的相位数据创建训练集、验证集和测试集。
- [CreateDataset_PhaseShift.py](CreateDataset_PhaseShift.py)：基于相位偏移量创建数据集。
- [metalens_model.py](metalens_model.py)：训练超透镜逆向设计网络。
- [metalens_phase_shift_model.py](metalens_phase_shift_model.py)：训练相位偏移预测网络。
- [metalens_design.py](metalens_design.py)：加载模型并生成最终超透镜设计结果。
- [Dataset3.cst](Dataset3.cst)：CST 仿真工程文件示例。

注意：仓库中有两个脚本文件名在实际路径里带有前导空格，使用时请以工作区中的真实文件名为准。

## 运行环境

代码主要面向 Windows + CST 仿真环境，部分脚本依赖以下组件：

- Python 3.12
- NumPy
- PyTorch
- Matplotlib
- SciPy
- tqdm
- CST Python API / 结果读取库

其中数据预处理脚本需要能够访问 CST 的 Python 库，以及本地 `.cst` 工程文件。

补充说明：[DataPrePro_PhaseShift.py](DataPrePro_PhaseShift.py) 的运行环境为 Python 3.6，执行前请切换到对应的解释器或虚拟环境。[metalens_model.py](metalens_model.py) + [metalens_phase_shift_model.py](metalens_phase_shift_model.py)训练环境为python3.12.


## 数据流程

推荐按以下顺序执行：

1. 运行 [DataPrePro_PhaseShift.py](DataPrePro_PhaseShift.py)，从 CST 工程中提取原始数据并生成平移后的相位数据。

2. 运行 [CreateDataset_ShiftedPhase.py](CreateDataset_ShiftedPhase.py)+ [CreateDataset_PhaseShift.py](CreateDataset_PhaseShift.py)把处理后的数据整理为训练集、验证集和测试集。

3. 运行 [metalens_phase_shift_model.py](metalens_phase_shift_model.py) 训练相位偏移预测模型。

4. 运行 [metalens_model.py](metalens_model.py) 训练逆向设计模型。

5. 运行 [metalens_design.py](metalens_design.py) 生成最终超透镜设计。

## 目录与输出

默认配置集中在 [config.py](config.py) 中，当前代码使用的根目录是 Windows 路径 `d:\CST output\CST 7`。运行后会在该根目录下生成或使用以下目录：

- `raw_data`：原始提取数据。
- `processed_data`：平移后的中间处理数据。
- `dataset_shifted_phase`：相位平移模型训练所需数据集。
- `dataset_phase_shift`：相位偏移模型训练所需数据集。
- `phase_shift_checkpoints`：相位偏移模型权重。
- `checkpoints` / `model_checkpoints`：逆向设计模型权重。
- `metalens_design`：最终设计结果与导出文件。
- `training_plots`：训练过程曲线图。

## 各脚本说明

### 1. 数据预处理 注意该git库并没有放置cst仿真源文件

[DataPrePro_PhaseShift.py](DataPrePro_PhaseShift.py) 会从 `Dataset3.cst` 中读取扫描参数组合，提取 `PHASE` 结果，并生成以下内容：

- 原始数据：`X_raw.npy`、`Y_raw.npy`、`Z_raw.npy`、`N_raw.npy`
- 处理后数据：`X_processed.npy`、`Y_processed.npy`、`Z_shifted.npy`、`phase_shift.npy`
- 报告文件：`phase_shift_report.txt`
- 可视化图：`phase_shift_visualization.png`

该脚本要求 CST 工程路径、CST Python 库路径和输出路径都能在本机正常访问。

### 2. 数据集构建

[CreateDataset_ShiftedPhase.py](CreateDataset_ShiftedPhase.py) 会读取 `processed_data` 中的文件，对结构参数和相位做归一化，并划分为训练集、验证集和测试集，输出：

- `X_train.npy`、`Y_train.npy`、`Z_train.npy`
- `X_val.npy`、`Y_val.npy`、`Z_val.npy`
- `X_test.npy`、`Y_test.npy`、`Z_test.npy`
- `norm_params.json`

[CreateDataset_PhaseShift.py](CreateDataset_PhaseShift.py) 会基于结构参数和相位偏移量生成另一套训练数据，用于相位偏移模型。

### 3. 相位偏移模型

[metalens_phase_shift_model.py](metalens_phase_shift_model.py) 训练一个输入为结构参数、输出为相位偏移量的网络。训练完成后会保存模型权重和训练曲线，并生成预测结果文件。

### 4. 逆向设计模型

[metalens_model.py](metalens_model.py) 训练超透镜逆向设计网络，目标是从相位分布预测结构参数 L 和 W。训练完成后会保存最佳模型、训练曲线和误差分布图。

### 5. 超透镜设计生成

[metalens_design.py](metalens_design.py) 会加载两个训练好的模型和归一化参数，先生成目标相位分布，再逐像素推理得到最终的 L/W 分布和相位偏移分布，并导出设计文件。
