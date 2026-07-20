
"""
基于相位偏移量的数据集制作代码 - CreateDataset_PhaseShift.py
环境: Python 3.12
功能: 基于相位偏移量创建神经网络训练数据集
"""

import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_processed_data(input_dir):
    """
    加载处理后的数据
    """
    print(f"加载处理后的数据: {input_dir}")
    
    X = np.load(os.path.join(input_dir, 'X_processed.npy'))
    phase_shift = np.load(os.path.join(input_dir, 'phase_shift.npy'))
    
    print(f"数据加载完成:")
    print(f"  X shape: {X.shape}")
    print(f"  phase_shift shape: {phase_shift.shape}")
    
    return X, phase_shift

def normalize_data(X, phase_shift):
    """
    数据归一化
    """
    X_min, X_max = X.min(axis=1, keepdims=True), X.max(axis=1, keepdims=True)
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)
    
    phase_min, phase_max = phase_shift.min(), phase_shift.max()
    phase_norm = 2 * (phase_shift - phase_min) / (phase_max - phase_min + 1e-8) - 1
    
    return X_norm, phase_norm, {'X_min': X_min, 'X_max': X_max, 'phase_min': phase_min, 'phase_max': phase_max}

def split_dataset(X, phase_shift, train_ratio=0.8):
    """
    划分训练集、验证集
    """
    n_samples = X.shape[1]
    np.random.seed(18)
    indices = np.random.permutation(n_samples)
    
    train_size = int(n_samples * train_ratio)
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    X_train, X_test = X[:, train_idx], X[:, test_idx]
    phase_train, phase_test = phase_shift[:, train_idx], phase_shift[:, test_idx]
    
    print(f"数据集划分: 训练集={len(train_idx)}, 测试集={len(test_idx)}")
    
    return X_train, X_test, phase_train, phase_test

def save_dataset(X_train, X_test, phase_train, phase_test, norm_params, output_dir='./dataset_phase_shift'):
    """
    保存处理后的数据集
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存各数据集
    np.save(f'{output_dir}/X_train.npy', X_train)
    np.save(f'{output_dir}/X_test.npy', X_test)
    np.save(f'{output_dir}/phase_shift_train.npy', phase_train)
    np.save(f'{output_dir}/phase_shift_test.npy', phase_test)
    
    # 保存归一化参数
    import json
    with open(f'{output_dir}/norm_params.json', 'w') as f:
        json.dump({
            'X_min': norm_params['X_min'].tolist(),
            'X_max': norm_params['X_max'].tolist(),
            'phase_min': float(norm_params['phase_min']),
            'phase_max': float(norm_params['phase_max'])
        }, f, indent=2)
    
    print(f"所有数据已保存到: {output_dir}")

def main():
    """主函数"""
    PROCESSED_DATA_DIR = r"d:\CST output\CST 7\dataprepro_phase_shift\processed_data"
    OUTPUT_DIR = r"d:\CST output\CST 7\dataset_phase_shift"
    
    print("=" * 60)
    print("开始基于相位偏移量的数据集制作流程")
    print("=" * 60)
    
    # 1. 加载处理后的数据
    print("加载处理后的数据...")
    X, phase_shift = load_processed_data(PROCESSED_DATA_DIR)
    
    # 2. 数据归一化
    print("进行数据归一化...")
    X_norm, phase_norm, norm_params = normalize_data(X, phase_shift)
    
    # 3. 划分数据集
    print("划分数据集...")
    X_train, X_test, phase_train, phase_test = split_dataset(X_norm, phase_norm, train_ratio=0.8)
    
    # 4. 保存数据
    print("保存数据集...")
    save_dataset(X_train, X_test, phase_train, phase_test, norm_params, OUTPUT_DIR)
    
    print("=" * 60)
    print("基于相位偏移量的数据集制作完成!")
    print(f"数据集已保存到: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()

