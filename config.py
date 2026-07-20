"""
超透镜设计配置文件 - config.py
环境: metalens (Python 3.12.4)
功能: 集中管理所有脚本的配置参数
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(r"d:\CST output\CST 7")

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"

# 数据集目录
DATASET_SHIFTED_PHASE = PROJECT_ROOT / "dataset_shifted_phase"
DATASET_PHASE_SHIFT = PROJECT_ROOT / "dataset_phase_shift"

# 模型检查点目录
MODEL_CHECKPOINTS = PROJECT_ROOT / "model_checkpoints"
PHASE_SHIFT_CHECKPOINTS = PROJECT_ROOT / "phase_shift_checkpoints"

# 输出目录
METALENS_DESIGN = PROJECT_ROOT / "metalens_design"
TRAINING_PLOTS = PROJECT_ROOT / "training_plots"

# 超透镜设计参数
NANOCELL_PERIOD = 450  # 纳米柱周期 (nm)
DIAMETER = 20.0  # 超透镜直径 (μm)
FOCAL_LENGTH = 30.0  # 焦距 (μm)
WAVELENGTHS = list(range(450, 650, 10))  # 工作波长范围 (nm)
RESOLUTION = int(DIAMETER * 1000 / NANOCELL_PERIOD)  # 根据周期自动计算分辨率

# 模型参数
INPUT_DIM = 1002  # 输入维度 (1002个频率点)
OUTPUT_DIM = 2  # 输出维度 (L, W)
PHASE_SHIFT_OUTPUT_DIM = 1  # 相位偏移模型输出维度

# 训练参数
BATCH_SIZE = 32
NUM_EPOCHS = 150
LEARNING_RATE = 0.001
PATIENCE = 20  # 早停耐心值

# 设备设置
try:
    import torch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    DEVICE = 'cpu'

# 日志配置
LOG_DIR = PROJECT_ROOT / "logs"
LOG_LEVEL = "INFO"

# 归一化参数文件路径
NORM_PARAMS_PATH = DATASET_SHIFTED_PHASE / "norm_params.json"

# 模型路径
BEST_MODEL_PATH = MODEL_CHECKPOINTS / "best_model.pth"
BEST_PHASE_SHIFT_MODEL_PATH = PHASE_SHIFT_CHECKPOINTS / "best_phase_shift_model.pth"

# 确保所有目录存在
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DATASET_SHIFTED_PHASE,
    DATASET_PHASE_SHIFT,
    MODEL_CHECKPOINTS,
    PHASE_SHIFT_CHECKPOINTS,
    METALENS_DESIGN,
    TRAINING_PLOTS,
    LOG_DIR
]:
    directory.mkdir(exist_ok=True)

# 打印配置信息
print("=" * 60)
print("超透镜设计配置")
print("=" * 60)
print(f"项目根目录: {PROJECT_ROOT}")
print(f"设备: {DEVICE}")
print(f"超透镜直径: {DIAMETER} μm")
print(f"焦距: {FOCAL_LENGTH} μm")
print(f"纳米柱周期: {NANOCELL_PERIOD} nm")
print(f"工作波长: {min(WAVELENGTHS)}-{max(WAVELENGTHS)} nm")
print(f"分辨率: {RESOLUTION}x{RESOLUTION}")
print("=" * 60)
