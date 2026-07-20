"""
超透镜深度学习模型 - metalens_model.py
环境: metalens (Python 3.12.4)
功能: 基于PyTorch 2.4.1的超透镜逆向设计神经网络模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import random
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')

def set_seed(seed=42):
    """设置所有随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 设置随机种子 - 必须在所有操作之前调用
set_seed(42)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

class SEAttention(nn.Module):
    """SE注意力模块"""
    def __init__(self, channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.view(b, c, 1)).view(b, c)
        y = self.fc(y)
        return x * y

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.LeakyReLU(0.01)
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.activation(out)
        return out

class MetaLensNet(nn.Module):
    """超透镜逆向设计神经网络 - 优化版"""
    def __init__(self, input_dim, output_dim=2):
        super(MetaLensNet, self).__init__()
        
        # 输入层 - 降低dropout率
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.02)
        )
        
        # SE注意力层
        self.attention = SEAttention(256, reduction=16)
        
        # 隐藏层 - 降低dropout率，释放模型容量
        self.hidden_layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.02),
            ResidualBlock(512, dropout=0.05),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.02),
            ResidualBlock(1024, dropout=0.05),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.02),
            ResidualBlock(512, dropout=0.05),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.02),
            ResidualBlock(256, dropout=0.05),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.02),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.02)
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.Tanh()
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.attention(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

def validate_data_dir(data_dir: str) -> bool:
    """
    验证数据集目录
    
    参数:
        data_dir: 数据集目录
    
    返回:
        bool: 目录是否有效
    """
    if not os.path.exists(data_dir):
        logging.error(f"数据集目录不存在: {data_dir}")
        return False
    required_files = [
        'X_train.npy', 'Y_train.npy', 'Z_train.npy',
        'X_val.npy', 'Y_val.npy', 'Z_val.npy',
        'X_test.npy', 'Y_test.npy', 'Z_test.npy'
    ]
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            logging.error(f"缺少必要文件: {file}")
            return False
    return True

def load_dataset(data_dir: str) -> Dict:
    """
    加载处理后的数据集
    
    参数:
        data_dir: 数据集目录
    
    返回:
        数据集字典
    """
    if not validate_data_dir(data_dir):
        raise ValueError(f"无效的数据集目录: {data_dir}")
    
    try:
        dataset = {}
        
        # 加载训练集
        dataset['train'] = {
            'X': np.load(os.path.join(data_dir, 'X_train.npy')),
            'Y': np.load(os.path.join(data_dir, 'Y_train.npy')),
            'Z': np.load(os.path.join(data_dir, 'Z_train.npy'))
        }
        
        # 加载验证集
        dataset['val'] = {
            'X': np.load(os.path.join(data_dir, 'X_val.npy')),
            'Y': np.load(os.path.join(data_dir, 'Y_val.npy')),
            'Z': np.load(os.path.join(data_dir, 'Z_val.npy'))
        }
        
        # 加载测试集
        dataset['test'] = {
            'X': np.load(os.path.join(data_dir, 'X_test.npy')),
            'Y': np.load(os.path.join(data_dir, 'Y_test.npy')),
            'Z': np.load(os.path.join(data_dir, 'Z_test.npy'))
        }
        
        # 验证数据形状
        for split in ['train', 'val', 'test']:
            if dataset[split]['X'].shape[1] != dataset[split]['Z'].shape[1]:
                logging.error(f"数据形状不匹配: {split}集 X={dataset[split]['X'].shape}, Z={dataset[split]['Z'].shape}")
                raise ValueError("数据形状不匹配")
        
        logging.info(f"数据集加载完成: 训练集={dataset['train']['X'].shape[1]}, 验证集={dataset['val']['X'].shape[1]}, 测试集={dataset['test']['X'].shape[1]}")
        return dataset
    except Exception as e:
        logging.error(f"加载数据集失败: {e}")
        raise

def worker_init_fn(worker_id):
    """DataLoader worker初始化函数，确保每个worker的随机种子一致"""
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def prepare_data(dataset: Dict, batch_size: int = 32) -> Dict:
    """
    准备数据加载器
    
    参数:
        dataset: 数据集字典
        batch_size: 批量大小
    
    返回:
        数据加载器字典
    """
    try:
        from torch.utils.data import TensorDataset, DataLoader
        
        loaders = {}
        g = torch.Generator()
        g.manual_seed(42)
        
        # 处理训练集
        train_X = torch.tensor(dataset['train']['Z'].T, dtype=torch.float32)
        train_y = torch.tensor(dataset['train']['X'].T, dtype=torch.float32)
        train_dataset = TensorDataset(train_X, train_y)
        loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                     worker_init_fn=worker_init_fn, generator=g)
        
        # 处理验证集
        val_X = torch.tensor(dataset['val']['Z'].T, dtype=torch.float32)
        val_y = torch.tensor(dataset['val']['X'].T, dtype=torch.float32)
        val_dataset = TensorDataset(val_X, val_y)
        loaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                   worker_init_fn=worker_init_fn, generator=g)
        
        # 处理测试集
        test_X = torch.tensor(dataset['test']['Z'].T, dtype=torch.float32)
        test_y = torch.tensor(dataset['test']['X'].T, dtype=torch.float32)
        test_dataset = TensorDataset(test_X, test_y)
        loaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                    worker_init_fn=worker_init_fn, generator=g)
        
        logging.info(f"数据加载器准备完成: batch_size={batch_size}")
        return loaders
    except Exception as e:
        logging.error(f"准备数据加载器失败: {e}")
        raise

def train_model(model: nn.Module, loaders: Dict, num_epochs: int = 200, 
                learning_rate: float = 0.0003, patience: int = 30, 
                checkpoint_dir: str = './checkpoints'):
    """
    训练模型
    
    参数:
        model: 模型
        loaders: 数据加载器
        num_epochs: 训练轮次
        learning_rate: 学习率
        patience: 早停 patience
        checkpoint_dir: 检查点保存目录
    """
    try:
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 设备选择
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        logging.info(f"使用设备: {device}")
        
        # 优化器 - 降低权重衰减，让模型学习更充分
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                              betas=(0.9, 0.95), eps=1e-7, weight_decay=1e-4)
        
        # 学习率调度 - 简单有效的ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=6, 
                                      min_lr=5e-7)
        
        # 改进损失函数 - 结合MSE和物理约束正则化
        criterion = nn.MSELoss()
        
        # 物理约束正则化系数 - 降低正则化强度，让损失更低
        lambda_smooth = 0.01
        
        # 早停
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        # 记录训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        logging.info("开始训练模型...")
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_mae = 0.0
            
            # 使用tqdm显示进度
            with tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch_X, batch_y in pbar:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    # 前向传播
                    outputs = model(batch_X)
                    mse_loss = criterion(outputs, batch_y)
                    
                    # 物理约束正则化: 输出参数平滑性约束
                    # L1正则化鼓励参数在合理范围内
                    l1_reg = torch.mean(torch.abs(outputs))
                    
                    # 组合损失
                    loss = mse_loss + lambda_smooth * l1_reg
                    
                    # 计算MAE
                    mae = torch.mean(torch.abs(outputs - batch_y))
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    # 梯度裁剪防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += mse_loss.item() * batch_X.size(0)
                    train_mae += mae.item() * batch_X.size(0)
                    
                    # 更新进度条
                    pbar.set_postfix({"loss": f"{mse_loss.item():.6f}", "mae": f"{mae.item():.6f}"})
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in loaders['val']:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    mae = torch.mean(torch.abs(outputs - batch_y))
                    val_loss += loss.item() * batch_X.size(0)
                    val_mae += mae.item() * batch_X.size(0)
            
            # 学习率调度 - ReduceLROnPlateau
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                logging.info(f"学习率从 {old_lr:.8f} 调整为 {new_lr:.8f}")
            
            # 计算平均损失
            train_loss /= len(loaders['train'].dataset)
            train_mae /= len(loaders['train'].dataset)
            val_loss /= len(loaders['val'].dataset)
            val_mae /= len(loaders['val'].dataset)
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            
            # 打印进度
            logging.info(f"Epoch {epoch+1}/{num_epochs}: ")
            logging.info(f"  训练损失: {train_loss:.6f}, 训练MAE: {train_mae:.6f}")
            logging.info(f"  验证损失: {val_loss:.6f}, 验证MAE: {val_mae:.6f}")
            
            # 保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss
                }, checkpoint_path)
                logging.info(f"检查点保存到: {checkpoint_path}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                # 保存最佳模型
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss
                }, best_model_path)
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    logging.info(f"早停触发: {patience}轮验证损失未改善")
                    break
        
        logging.info("模型训练完成!")
        return model, history
    except Exception as e:
        logging.error(f"训练模型失败: {e}")
        raise

def evaluate_model(model: nn.Module, loaders: Dict):
    """
    评估模型
    
    参数:
        model: 模型
        loaders: 数据加载器
    """
    try:
        # 设备选择
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # 损失函数
        criterion = nn.MSELoss()
        
        model.eval()
        test_loss = 0.0
        test_mae = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in loaders['test']:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                mae = torch.mean(torch.abs(outputs - batch_y))
                test_loss += loss.item() * batch_X.size(0)
                test_mae += mae.item() * batch_X.size(0)
        
        test_loss /= len(loaders['test'].dataset)
        test_mae /= len(loaders['test'].dataset)
        
        logging.info("模型评估结果:")
        logging.info(f"  测试损失: {test_loss:.6f}")
        logging.info(f"  测试MAE: {test_mae:.6f}")
        
        return test_loss, test_mae
    except Exception as e:
        logging.error(f"评估模型失败: {e}")
        raise

def smooth_curve(data: List, window_size: int = 5) -> List:
    """
    使用移动平均平滑曲线
    
    参数:
        data: 原始数据列表
        window_size: 滑动窗口大小
        
    返回:
        平滑后的数据列表
    """
    if len(data) < window_size:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        smoothed.append(np.mean(data[start:i+1]))
    return smoothed

def plot_training_curves(history: Dict, save_dir: str = './training_plots', smooth_window: int = 5):
    """
    绘制训练曲线（包含原始曲线和平滑曲线）
    
    参数:
        history: 训练历史字典
        save_dir: 图片保存目录
        smooth_window: 平滑窗口大小
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        
        # 平滑处理数据
        train_loss_smooth = smooth_curve(history['train_loss'], smooth_window)
        val_loss_smooth = smooth_curve(history['val_loss'], smooth_window)
        train_mae_smooth = smooth_curve(history['train_mae'], smooth_window)
        val_mae_smooth = smooth_curve(history['val_mae'], smooth_window)
        
        # 绘制损失曲线 - 双图对比（原始+平滑）
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='训练MSE(原始)', linewidth=1, alpha=0.4, color='#1f77b4')
        plt.plot(history['val_loss'], label='验证MSE(原始)', linewidth=1, alpha=0.4, color='#ff7f0e')
        plt.plot(train_loss_smooth, label='训练MSE(平滑)', linewidth=2, color='#1f77b4')
        plt.plot(val_loss_smooth, label='验证MSE(平滑)', linewidth=2, color='#ff7f0e')
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('MSE Loss', fontsize=20)
        plt.title(f'训练和验证MSE曲线 ', fontsize=20, fontweight='bold')
        plt.legend(fontsize=20)
        plt.grid(True, alpha=0.3)
        
        # 绘制MAE曲线
        plt.subplot(1, 2, 2)
        plt.plot(history['train_mae'], label='训练MAE(原始)', linewidth=1, alpha=0.4, color='#2ca02c')
        plt.plot(history['val_mae'], label='验证MAE(原始)', linewidth=1, alpha=0.4, color='#d62728')
        plt.plot(train_mae_smooth, label='训练MAE(平滑)', linewidth=2, color='#2ca02c')
        plt.plot(val_mae_smooth, label='验证MAE(平滑)', linewidth=2, color='#d62728')
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('MAE Loss', fontsize=20)
        plt.title(f'训练和验证MAE曲线 ', fontsize=20, fontweight='bold')
        plt.legend(fontsize=20)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        loss_curve_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"训练曲线已保存到: {loss_curve_path}")
        
        return loss_curve_path
    except Exception as e:
        logging.error(f"绘制训练曲线失败: {e}")
        raise

def plot_prediction_scatter(model: nn.Module, loaders: Dict, save_dir: str = './training_plots'):
    """
    绘制预测值与真实值散点图
    
    参数:
        model: 训练好的模型
        loaders: 数据加载器
        save_dir: 图片保存目录
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        all_true = []
        all_pred = []
        
        with torch.no_grad():
            for batch_X, batch_y in loaders['test']:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                all_true.append(batch_y.cpu().numpy())
                all_pred.append(outputs.cpu().numpy())
        
        all_true = np.vstack(all_true)
        all_pred = np.vstack(all_pred)
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        output_names = ['长度 L (nm)', '宽度 W (nm)']
        
        for i in range(2):
            ax = axes[i]
            true_vals = all_true[:, i]
            pred_vals = all_pred[:, i]
            
            ax.scatter(true_vals, pred_vals, alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
            
            min_val = min(true_vals.min(), pred_vals.min())
            max_val = max(true_vals.max(), pred_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')
            
            mae = np.mean(np.abs(true_vals - pred_vals))
            rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
            
            ax.set_xlabel(f'真实值 {output_names[i]}', fontsize=20)
            ax.set_ylabel(f'预测值 {output_names[i]}', fontsize=20)
            ax.set_title(f'{output_names[i]}预测结果\nMAE = {mae:.4f}, RMSE = {rmse:.4f}', 
                         fontsize=20, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        scatter_path = os.path.join(save_dir, 'prediction_scatter.png')
        plt.savefig(scatter_path, dpi=600, bbox_inches='tight')
        plt.close()
        logging.info(f"预测散点图已保存到: {scatter_path}")
        
        return scatter_path
    except Exception as e:
        logging.error(f"绘制预测散点图失败: {e}")
        raise

def plot_error_distribution(model: nn.Module, loaders: Dict, save_dir: str = './training_plots'):
    """
    绘制预测误差分布直方图
    
    参数:
        model: 训练好的模型
        loaders: 数据加载器
        save_dir: 图片保存目录
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        all_true = []
        all_pred = []
        
        with torch.no_grad():
            for batch_X, batch_y in loaders['test']:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                all_true.append(batch_y.cpu().numpy())
                all_pred.append(outputs.cpu().numpy())
        
        all_true = np.vstack(all_true)
        all_pred = np.vstack(all_pred)
        errors = all_pred - all_true
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        output_names = ['长度 L (nm)', '宽度 W (nm)']
        
        for i in range(2):
            ax = axes[i]
            err = errors[:, i]
            
            n, bins, patches = ax.hist(err, bins=30, edgecolor='white', alpha=0.7)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.axvline(x=np.mean(err), color='green', linestyle='-', linewidth=2, 
                       label=f'均值: {np.mean(err):.4f}')
            
            ax.set_xlabel('预测误差 (预测值 - 真实值)', fontsize=20)
            ax.set_ylabel('样本数量', fontsize=20)
            ax.set_title(f'{output_names[i]}误差分布\nstd = {np.std(err):.4f}', 
                         fontsize=20, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        error_dist_path = os.path.join(save_dir, 'error_distribution.png')
        plt.savefig(error_dist_path, dpi=600, bbox_inches='tight')
        plt.close()
        logging.info(f"误差分布图已保存到: {error_dist_path}")
        
        return error_dist_path
    except Exception as e:
        logging.error(f"绘制误差分布图失败: {e}")
        raise

def main():
    """主函数"""
    # 配置参数 - 优化版
    DATA_DIR = r"d:\CST output\CST 7\dataset_shifted_phase"
    CHECKPOINT_DIR = r"d:\CST output\CST 7\checkpoints"
    PLOT_DIR = r"d:\CST output\CST 7\training_plots"
    BATCH_SIZE = 64
    NUM_EPOCHS = 600
    LEARNING_RATE = 0.0003
    
    # 打印流程开始的分隔线和提示信息
    logging.info("=" * 60)
    logging.info("开始模型训练流程")
    logging.info("=" * 60)
    
    try:
        # 1. 加载数据集
        logging.info("加载数据集...")
        dataset = load_dataset(DATA_DIR)
        
        # 2. 准备数据加载器
        logging.info("准备数据加载器...")
        loaders = prepare_data(dataset, batch_size=BATCH_SIZE)
        
        # 3. 创建模型
        input_dim = dataset['train']['Z'].shape[0]  # 输入维度为频率点数量
        logging.info(f"创建模型: 输入维度={input_dim}")
        model = MetaLensNet(input_dim=input_dim, output_dim=2)
        
        # 4. 训练模型
        logging.info("训练模型...")
        model, history = train_model(model, loaders, num_epochs=NUM_EPOCHS, 
                                    learning_rate=LEARNING_RATE, checkpoint_dir=CHECKPOINT_DIR)
        
        # 5. 加载最佳模型
        logging.info("加载最佳模型进行评估...")
        best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
        checkpoint = torch.load(best_model_path, weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # 6. 创建只包含验证集的loaders子集用于评估和可视化（因为现在是8:2划分，验证集是20%的完整数据）
        val_only_loaders = {
            'val': loaders['val'],
            'test': loaders['val']  # 将评估和可视化指向完整的验证集
        }
        
        # 7. 评估模型 - 在完整验证集上进行
        logging.info("在完整验证集(20%数据)上评估模型...")
        val_loss, val_mae = evaluate_model(model, val_only_loaders)
        
        # 8. 可视化训练结果
        logging.info("生成训练结果可视化...")
        plot_training_curves(history, save_dir=PLOT_DIR)
        plot_prediction_scatter(model, val_only_loaders, save_dir=PLOT_DIR)
        plot_error_distribution(model, val_only_loaders, save_dir=PLOT_DIR)
        logging.info("所有可视化图表已生成完成!")
        
        logging.info("=" * 60)
        logging.info("模型训练流程完成! (数据集划分: 训练集80% / 验证集20%)")
        logging.info(f"可视化图表保存在: {PLOT_DIR}")
        logging.info("=" * 60)
    except Exception as e:
        logging.error(f"训练流程失败: {e}")
        raise

if __name__ == "__main__":
    main()
