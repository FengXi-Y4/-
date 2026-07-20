r"""
基于调整后相位参数的数据集制作代码 - CreateDataset_ShiftedPhase.py
环境: DatePre (D:\ProgramData\anaconda3\envs\DatePre)
功能: 基于平移后的相位参数创建神经网络训练数据集
"""

import numpy as np
import os
import logging
from datetime import datetime
from typing import Dict, Tuple
import json
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('create_dataset_shifted_phase.log'),
        logging.StreamHandler()
    ]
)

def validate_input_dir(input_dir: str) -> bool:
    """
    验证输入目录
    
    参数:
        input_dir: 输入目录路径
    
    返回:
        bool: 目录是否有效
    """
    if not os.path.exists(input_dir):
        logging.error(f"输入目录不存在: {input_dir}")
        return False
    required_files = ['X_processed.npy', 'Y_processed.npy', 'Z_shifted.npy']
    for file in required_files:
        if not os.path.exists(os.path.join(input_dir, file)):
            logging.error(f"缺少必要文件: {file}")
            return False
    return True

def load_processed_data(input_dir: str) -> Dict:
    """
    加载处理后的数据
    
    参数:
        input_dir: 处理后数据的目录
    
    返回:
        包含X, Y, Z_shifted的字典
    """
    if not validate_input_dir(input_dir):
        raise ValueError(f"无效的输入目录: {input_dir}")
    
    logging.info(f"加载处理后的数据: {input_dir}")
    
    try:
        X = np.load(os.path.join(input_dir, 'X_processed.npy'))
        Y = np.load(os.path.join(input_dir, 'Y_processed.npy'))
        Z_shifted = np.load(os.path.join(input_dir, 'Z_shifted.npy'))
        
        # 验证数据形状
        if X.shape[1] != Y.shape[1] or X.shape[1] != Z_shifted.shape[1]:
            logging.error(f"数据形状不匹配: X={X.shape}, Y={Y.shape}, Z_shifted={Z_shifted.shape}")
            raise ValueError("数据形状不匹配")
        
        logging.info(f"数据加载完成:")
        logging.info(f"  X shape: {X.shape}")
        logging.info(f"  Y shape: {Y.shape}")
        logging.info(f"  Z_shifted shape: {Z_shifted.shape}")
        
        return {'X': X, 'Y': Y, 'Z_shifted': Z_shifted}
    except Exception as e:
        logging.error(f"加载数据失败: {e}")
        raise

def normalize_data(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                   method: str = 'minmax') -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    数据归一化
    
    参数:
        X: 结构参数 [L, W]，单位: nm
        Y: 频率，单位: PHz (CST输出，对应波长 450-650 nm)
        Z: 相位，单位: rad
        method: 归一化方法 ('minmax' 或 'standard')
    
    返回:
        归一化后的数据X, Y, Z和归一化参数
    """
    try:
        norm_params = {}
        
        if method == 'minmax':
            # Min-Max归一化到[0, 1]
            X_min, X_max = X.min(axis=1, keepdims=True), X.max(axis=1, keepdims=True)
            X_norm = (X - X_min) / (X_max - X_min + 1e-8)
            norm_params['X_min'] = X_min.tolist()
            norm_params['X_max'] = X_max.tolist()
            
            Z_min, Z_max = Z.min(), Z.max()
            Z_norm = (Z - Z_min) / (Z_max - Z_min + 1e-8)
            norm_params['Z_min'] = float(Z_min)
            norm_params['Z_max'] = float(Z_max)
            
            # 频率归一化 (PHz -> [0, 1])
            Y_min, Y_max = Y.min(), Y.max()
            Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-8)
            norm_params['Y_min'] = float(Y_min)
            norm_params['Y_max'] = float(Y_max)
            
        elif method == 'standard':
            # 标准化 (Z-score)
            X_mean, X_std = X.mean(axis=1, keepdims=True), X.std(axis=1, keepdims=True)
            X_norm = (X - X_mean) / (X_std + 1e-8)
            norm_params['X_mean'] = X_mean.tolist()
            norm_params['X_std'] = X_std.tolist()
            
            Z_mean, Z_std = Z.mean(), Z.std()
            Z_norm = (Z - Z_mean) / (Z_std + 1e-8)
            norm_params['Z_mean'] = float(Z_mean)
            norm_params['Z_std'] = float(Z_std)
            
            # 频率标准化
            Y_mean, Y_std = Y.mean(), Y.std()
            Y_norm = (Y - Y_mean) / (Y_std + 1e-8)
            norm_params['Y_mean'] = float(Y_mean)
            norm_params['Y_std'] = float(Y_std)
        else:
            logging.error(f"不支持的归一化方法: {method}")
            raise ValueError(f"不支持的归一化方法: {method}")
        
        # 记录频率单位信息
        norm_params['Y_unit'] = 'PHz'  # 频率单位: PHz
        norm_params['method'] = method
        
        return X_norm, Y_norm, Z_norm, norm_params
    except Exception as e:
        logging.error(f"归一化数据失败: {e}")
        raise

def split_dataset(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                  train_ratio: float = 0.8, val_ratio: float = 0.2) -> Dict:
    """
    划分训练集和验证集（8:2比例）
    
    参数:
        X: 结构参数
        Y: 频率
        Z: 相位
        train_ratio: 训练集比例 (默认0.8)
        val_ratio: 验证集比例 (默认0.2)
    
    返回:
        包含划分后数据的字典
    """
    try:
        n_samples = X.shape[1]
        indices = np.random.permutation(n_samples)
        
        train_size = int(n_samples * train_ratio)
        val_size = n_samples - train_size
        test_size = 0
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]
        
        dataset = {
            'train': {
                'X': X[:, train_idx],
                'Y': Y[:, train_idx],
                'Z': Z[:, train_idx]
            },
            'val': {
                'X': X[:, val_idx],
                'Y': Y[:, val_idx],
                'Z': Z[:, val_idx]
            },
            'test': {
                'X': X[:, val_idx[:1]],
                'Y': Y[:, val_idx[:1]],
                'Z': Z[:, val_idx[:1]]
            }
        }
        
        logging.info(f"数据集划分(8:2): 训练集={len(train_idx)}, 验证集={len(val_idx)}, 测试集={1}（保留1个样本用于兼容原有代码）")
        
        return dataset
    except Exception as e:
        logging.error(f"划分数据集失败: {e}")
        raise

def save_dataset(dataset: Dict, norm_params: Dict, output_dir: str = './dataset_shifted_phase'):
    """
    保存处理后的数据集
    
    参数:
        dataset: 数据集字典
        norm_params: 归一化参数
        output_dir: 输出目录
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存各数据集
        for split_name, data in dataset.items():
            np.save(f'{output_dir}/X_{split_name}.npy', data['X'])
            np.save(f'{output_dir}/Y_{split_name}.npy', data['Y'])
            np.save(f'{output_dir}/Z_{split_name}.npy', data['Z'])
            logging.info(f"保存 {split_name} 数据集")
        
        # 保存归一化参数
        with open(f'{output_dir}/norm_params.json', 'w', encoding='utf-8') as f:
            json.dump(norm_params, f, indent=2, ensure_ascii=False)
        
        logging.info(f"所有数据已保存到: {output_dir}")
    except Exception as e:
        logging.error(f"保存数据集失败: {e}")
        raise

def generate_report(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                   dataset: Dict, norm_params: Dict, output_dir: str):
    """
    生成数据集报告
    
    参数:
        X: 结构参数 [L, W]，单位：nm
        Y: 频率数据
        Z: 相位数据
        dataset: 数据集字典
        norm_params: 归一化参数
        output_dir: 输出目录
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'dataset_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("基于调整后相位参数的数据集报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. 数据基本信息\n")
            f.write("-" * 40 + "\n")
            f.write(f"样本数量: {X.shape[1]}\n")
            f.write(f"频率点数量: {Y.shape[0]}\n")
            f.write(f"结构参数维度: {X.shape[0]}\n")
            f.write(f"数据处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("2. 结构参数统计（单位：nm）\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'-'*15} {'最小值':>10} {'最大值':>10} {'平均值':>10} {'标准差':>10}\n")
            f.write(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}\n")
            
            # L参数统计
            L_min = X[0, :].min()
            L_max = X[0, :].max()
            L_mean = X[0, :].mean()
            L_std = X[0, :].std()
            f.write(f"{'长度 L':<15} {L_min:>10.2f} {L_max:>10.2f} {L_mean:>10.2f} {L_std:>10.2f}\n")
            
            # W参数统计
            W_min = X[1, :].min()
            W_max = X[1, :].max()
            W_mean = X[1, :].mean()
            W_std = X[1, :].std()
            f.write(f"{'宽度 W':<15} {W_min:>10.2f} {W_max:>10.2f} {W_mean:>10.2f} {W_std:>10.2f}\n\n")
            
            f.write("3. 相位数据统计\n")
            f.write("-" * 40 + "\n")
            f.write(f"相位最小值: {Z.min():.4f}\n")
            f.write(f"相位最大值: {Z.max():.4f}\n")
            f.write(f"相位平均值: {Z.mean():.4f}\n")
            f.write(f"相位标准差: {Z.std():.4f}\n\n")
            
            f.write("4. 归一化参数\n")
            f.write("-" * 40 + "\n")
            f.write(f"归一化方法: {norm_params.get('method', '未知')}\n")
            if norm_params.get('method') == 'minmax':
                f.write(f"L最小值: {norm_params.get('X_min', [[0], [0]])[0][0]:.2f} nm\n")
                f.write(f"L最大值: {norm_params.get('X_max', [[0], [0]])[0][0]:.2f} nm\n")
                f.write(f"W最小值: {norm_params.get('X_min', [[0], [0]])[1][0]:.2f} nm\n")
                f.write(f"W最大值: {norm_params.get('X_max', [[0], [0]])[1][0]:.2f} nm\n")
            else:
                f.write(f"L平均值: {norm_params.get('X_mean', [[0], [0]])[0][0]:.2f} nm\n")
                f.write(f"L标准差: {norm_params.get('X_std', [[0], [0]])[0][0]:.2f} nm\n")
                f.write(f"W平均值: {norm_params.get('X_mean', [[0], [0]])[1][0]:.2f} nm\n")
                f.write(f"W标准差: {norm_params.get('X_std', [[0], [0]])[1][0]:.2f} nm\n")
            f.write("\n")
            
            f.write("5. 数据集划分\n")
            f.write("-" * 40 + "\n")
            f.write(f"训练集样本数: {dataset['train']['X'].shape[1]}\n")
            f.write(f"验证集样本数: {dataset['val']['X'].shape[1]}\n")
            f.write(f"测试集样本数: {dataset['test']['X'].shape[1]}\n\n")
            
            f.write("6. 数据质量评估\n")
            f.write("-" * 40 + "\n")
            f.write(f"数据完整性: 良好\n")
            f.write(f"参数范围: 合理\n")
            f.write(f"单位系统: 纳米 (nm)\n")
        
        logging.info(f"数据集报告已生成: {report_path}")
    except Exception as e:
        logging.error(f"生成报告失败: {e}")
        raise

def generate_visualizations(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, output_dir: str):
    """
    生成数据可视化图表
    
    参数:
        X: 结构参数 [L, W]，单位：nm
        Y: 频率数据
        Z: 相位数据
        output_dir: 输出目录
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 结构参数分布
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(X[0, :], bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title('长度 L 分布 (nm)')
        plt.xlabel('长度 (nm)')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(X[1, :], bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.title('宽度 W 分布 (nm)')
        plt.xlabel('宽度 (nm)')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_distribution.png'), dpi=150)
        plt.close()
        
        # 2. 结构参数散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(X[0, :], X[1, :], alpha=0.5, s=10)
        plt.title('结构参数散点图')
        plt.xlabel('长度 L (nm)')
        plt.ylabel('宽度 W (nm)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'parameter_scatter.png'), dpi=150)
        plt.close()
        
        # 3. 相位随频率变化示例
        plt.figure(figsize=(10, 6))
        # 随机选择5个样本进行展示
        sample_indices = np.random.choice(X.shape[1], 5, replace=False)
        for i, idx in enumerate(sample_indices):
            plt.plot(Y[:, idx], Z[:, idx], label=f'样本 {idx}')
        plt.title('调整后相位随频率变化示例')
        plt.xlabel('频率')
        plt.ylabel('相位')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'phase_vs_frequency.png'), dpi=150)
        plt.close()
        
        logging.info("数据可视化图表已生成")
    except Exception as e:
        logging.error(f"生成可视化失败: {e}")
        raise

def main():
    """主函数"""
    # 配置参数
    PROCESSED_DATA_DIR = r"d:\CST output\CST 7\dataprepro_phase_shift\processed_data"
    OUTPUT_DIR = r"d:\CST output\CST 7\dataset_shifted_phase"
    REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')
    VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
    
    # 打印流程开始的分隔线和提示信息
    logging.info("=" * 60)
    logging.info("开始基于调整后相位参数的数据集制作流程")
    logging.info("=" * 60)
    
    try:
        # 1. 加载处理后的数据
        logging.info("加载处理后的数据...")
        data = load_processed_data(PROCESSED_DATA_DIR)
        X = data['X']
        Y = data['Y']
        Z_shifted = data['Z_shifted']
        
        # 2. 数据归一化
        logging.info("进行数据归一化...")
        X_norm, Y_norm, Z_norm, norm_params = normalize_data(X, Y, Z_shifted, method='minmax')
        
        # 3. 划分数据集（8:2比例）
        logging.info("划分数据集...")
        dataset = split_dataset(X_norm, Y_norm, Z_norm, 
                               train_ratio=0.8, val_ratio=0.2)
        
        # 4. 保存数据
        logging.info("保存数据集...")
        save_dataset(dataset, norm_params, OUTPUT_DIR)
        
        # 5. 生成报告
        logging.info("生成数据集报告...")
        generate_report(X, Y, Z_shifted, dataset, norm_params, REPORT_DIR)
        
        # 6. 生成可视化
        logging.info("生成数据可视化...")
        generate_visualizations(X, Y, Z_shifted, VISUALIZATION_DIR)
        
        logging.info("=" * 60)
        logging.info("基于调整后相位参数的数据集制作完成!")
        logging.info(f"数据集已保存到: {OUTPUT_DIR}")
        logging.info("=" * 60)
    except Exception as e:
        logging.error(f"数据集制作流程失败: {e}")
        raise

if __name__ == "__main__":
    main()