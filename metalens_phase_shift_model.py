
"""
超透镜相位偏移预测模型 - metalens_phase_shift_model.py
环境: Python 3.12
功能: 基于PyTorch的超透镜相位偏移预测神经网络模型
"""

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

n_features = 2
out_features = 1
tr_rate = 0.8
training_step = 2000
lr = 0.0001
step_size = 100
gamma = 0.5
dropoutrate = 0.00000
batch_size = 128

# 模型保存路径 - 与 metalens_design.py 一致
CHECKPOINT_DIR = r"d:\CST output\CST 7\phase_shift_checkpoints"
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, 'best_phase_shift_model.pth')

# 输出文件保存目录
OUTPUT_DIR = r"d:\CST output\CST 7\phase_shift_outputs"
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
LOSS_FILE = os.path.join(OUTPUT_DIR, 'LossF.txt')
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'Train.txt')
TEST_FILE = os.path.join(OUTPUT_DIR, 'Test.txt')
ALL_FILE = os.path.join(OUTPUT_DIR, 'All.txt')
TRAINING_CURVES_FILE = os.path.join(PLOT_DIR, 'training_curves.png')
PREDICTION_SCATTER_FILE = os.path.join(PLOT_DIR, 'prediction_scatter.png')

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class MetaLensPhaseShiftNet(torch.nn.Module):
    """超透镜相位偏移预测神经网络
    优化: 使用LayerNorm替代BatchNorm，提高训练稳定性
    并移除输出层附近的归一化层，避免梯度消失
    """
    def __init__(self, input_dim=2, output_dim=1):
        super(MetaLensPhaseShiftNet, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.05),
            nn.LayerNorm(128),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.05),
            nn.LayerNorm(256),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.05),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.05),
            
            nn.Linear(64, output_dim)
        )
    
    def forward(self, inputs):
        out = self.l1(inputs)
        return out

def load_dataset(data_dir):
    """
    加载数据集
    """
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    phase_train = np.load(os.path.join(data_dir, 'phase_shift_train.npy'))
    phase_test = np.load(os.path.join(data_dir, 'phase_shift_test.npy'))
    
    return X_train, X_test, phase_train, phase_test

def denormalize_phase(phase_norm, phase_min, phase_max):
    """
    将归一化的相位值[-1, 1]反归一化到原始范围[phase_min, phase_max]
    """
    return (phase_norm + 1) * (phase_max - phase_min) / 2 + phase_min

def load_norm_params(data_dir):
    """
    加载归一化参数
    """
    import json
    with open(os.path.join(data_dir, 'norm_params.json'), 'r') as f:
        params = json.load(f)
    return params['phase_min'], params['phase_max']

def train():
    """
    训练模型
    """
    DATA_DIR = r"d:\CST output\CST 7\dataset_phase_shift"
    
    print("=" * 60)
    print("开始相位偏移模型训练")
    print("=" * 60)
    
    print("加载数据集...")
    X_train, X_test, phase_train, phase_test = load_dataset(DATA_DIR)
    
    X_train = X_train.T
    X_test = X_test.T
    phase_train = phase_train.T
    phase_test = phase_test.T
    
    print(f"训练集: X={X_train.shape}, y={phase_train.shape}")
    print(f"测试集: X={X_test.shape}, y={phase_test.shape}")
    
    X_train = torch.from_numpy(X_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    phase_train = torch.from_numpy(phase_train).float().to(device)
    phase_test = torch.from_numpy(phase_test).float().to(device)
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("创建模型...")
    model = MetaLensPhaseShiftNet(input_dim=n_features, output_dim=out_features).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    loss_func = torch.nn.MSELoss().to(device)
    
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    print("开始训练...")
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }
    PltOut = np.zeros(shape=(training_step, 3))
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 50
    early_stop_counter = 0
    
    for step in range(training_step):
        model.train()
        M_train = len(X_train)
        train_epoch_loss = 0.0
        train_epoch_mae = 0.0
        
        permutation = torch.randperm(M_train)
        
        with tqdm(np.arange(0, M_train, batch_size), desc=f"Epoch {step+1}/{training_step}") as pbar:
            for index in pbar:
                L = index
                R = min(M_train, index + batch_size)
                indices = permutation[L:R]
                
                train_pre = model(X_train[indices])
                train_real = phase_train[indices]
                train_loss = loss_func(train_pre, train_real)
                train_mae = torch.mean(torch.abs(train_pre - train_real))
                
                train_epoch_loss += train_loss.item() * (R - L)
                train_epoch_mae += train_mae.item() * (R - L)
                
                optimizer.zero_grad()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_pre = model(X_test)
                val_real = phase_test.reshape(len(phase_test), out_features)
                val_loss = loss_func(val_pre, val_real)
                val_mae = torch.mean(torch.abs(val_pre - val_real))
            
            pbar.set_postfix(train_loss=float(train_epoch_loss/(R)), train_mae=float(train_epoch_mae/(R)),
                           val_loss=float(val_loss.data), val_mae=float(val_mae.data))
        
        train_epoch_loss /= M_train
        train_epoch_mae /= M_train
        
        history['train_loss'].append(train_epoch_loss)
        history['val_loss'].append(float(val_loss.data))
        history['train_mae'].append(train_epoch_mae)
        history['val_mae'].append(float(val_mae.data))
        
        scheduler.step(val_loss)
        
        PltOut[step, 0:] = [step, float(val_loss.data), train_epoch_loss]
        np.savetxt(LOSS_FILE, PltOut, fmt='%s', delimiter=',')
        
        if float(val_loss.data) < best_val_loss:
            best_val_loss = float(val_loss.data)
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            print(f"早停: 验证损失{patience}轮没有改善")
            break
    
    print(f"最佳验证损失: {best_val_loss:.6f}")
    torch.save(best_model_state, MODEL_SAVE_PATH)
    print(f"最佳模型已保存到: {MODEL_SAVE_PATH}")
    
    plot_training_curves(history, PLOT_DIR, TRAINING_CURVES_FILE)
    
    print("=" * 60)
    print("训练完成!")
    print("=" * 60)
    
    return history, model

def predict():
    """
    使用模型进行预测
    """
    DATA_DIR = r"d:\CST output\CST 7\dataset_phase_shift"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("开始预测")
    print("=" * 60)
    
    phase_min, phase_max = load_norm_params(DATA_DIR)
    
    X_train, X_test, phase_train, phase_test = load_dataset(DATA_DIR)
    
    X = np.concatenate([X_train, X_test], axis=1)
    
    X_train = X_train.T
    X_test = X_test.T
    X = X.T
    
    X_train = torch.from_numpy(X_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    X = torch.from_numpy(X).float().to(device)
    
    model = MetaLensPhaseShiftNet(input_dim=n_features, output_dim=out_features).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    
    model.eval()
    
    train_pre = model(X_train)
    train_pre = train_pre.detach().cpu().numpy()
    train_pre = denormalize_phase(train_pre, phase_min, phase_max)
    np.savetxt(TRAIN_FILE, train_pre, fmt='%s', delimiter=',')
    
    test_pre = model(X_test)
    test_pre = test_pre.detach().cpu().numpy()
    test_pre = denormalize_phase(test_pre, phase_min, phase_max)
    np.savetxt(TEST_FILE, test_pre, fmt='%s', delimiter=',')
    
    all_pre = model(X)
    all_pre = all_pre.detach().cpu().numpy()
    all_pre = denormalize_phase(all_pre, phase_min, phase_max)
    np.savetxt(ALL_FILE, all_pre, fmt='%s', delimiter=',')
    
    print("预测结果已保存:")
    print(f"  {TRAIN_FILE} - 训练集预测结果")
    print(f"  {TEST_FILE} - 测试集预测结果")
    print(f"  {ALL_FILE} - 全部数据预测结果")
    
    print("=" * 60)
    print("预测完成!")
    print("=" * 60)

def plot_training_curves(history, plot_dir, save_path):
    """
    绘制训练曲线 - 与 metalens_model.py 样式完全一致
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    
    # 绘制损失曲线 (1x2 布局)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练MSE ', linewidth=2)
    plt.plot(history['val_loss'], label='验证MSE', linewidth=2)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('MSE Loss', fontsize=20)
    plt.title('训练和验证MSE曲线', fontsize=20, fontweight='bold')
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    
    # 绘制MAE曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='训练MAE', linewidth=2)
    plt.plot(history['val_mae'], label='验证MAE', linewidth=2)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('MAE Loss', fontsize=20)
    plt.title('训练和验证MAE曲线', fontsize=20, fontweight='bold')
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到: {save_path}")

def plot_prediction_scatter(model, X_test, y_test, save_path, phase_min, phase_max):
    """
    绘制预测值与真实值散点图 - 与 metalens_model.py 样式一致
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        X_test_tensor = X_test.to(device)
        y_pred = model(X_test_tensor)
        y_pred = y_pred.cpu().numpy()
        y_true = y_test.cpu().numpy()
    
    y_pred = denormalize_phase(y_pred, phase_min, phase_max)
    y_true = denormalize_phase(y_true, phase_min, phase_max)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    
    plt.figure(figsize=(10, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    plt.xlabel('真实值 相位偏移 (rad)', fontsize=20)
    plt.ylabel('预测值 相位偏移 (rad)', fontsize=20)
    plt.title(f'相位偏移预测结果\nMAE = {mae:.4f}, RMSE = {rmse:.4f}', 
                fontsize=20, fontweight='bold')
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"预测散点图已保存到: {save_path}")

def main():
    """主函数"""
    DATA_DIR = r"d:\CST output\CST 7\dataset_phase_shift"
    
    print(f"使用设备: {device}")
    
    phase_min, phase_max = load_norm_params(DATA_DIR)
    
    history, model = train()
    
    X_train, X_test, phase_train, phase_test = load_dataset(DATA_DIR)
    X_test = X_test.T
    phase_test = phase_test.T
    X_test = torch.from_numpy(X_test).float().to(device)
    phase_test = torch.from_numpy(phase_test).float().to(device)
    
    plot_prediction_scatter(model, X_test, phase_test, PREDICTION_SCATTER_FILE, phase_min, phase_max)
    
    predict()

if __name__ == "__main__":
    main()

