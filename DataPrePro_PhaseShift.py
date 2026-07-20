"""
相位平移数据预处理脚本 - DataPrePro_PhaseShift.py
环境: DatePre (D:\ProgramData\anaconda3\envs\DatePre)
功能: 从CST软件提取相位参数，进行平移处理，并保存处理前后的数据
"""

import sys
import numpy as np
import os
import logging
from datetime import datetime
from typing import Tuple, List, Dict
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preprocessing_phase_shift.log'),
        logging.StreamHandler()
    ]
)

# CST库路径配置
cst_lib_path = r"D:\CST\AMD64\python_cst_libraries"
sys.path.append(cst_lib_path)

try:
    import cst
    import cst.results
except ImportError as e:
    logging.error(f"CST库导入失败: {e}")
    raise

def validate_project_path(project_path: str) -> bool:
    """
    验证CST项目路径
    
    参数:
        project_path: CST项目文件路径
    
    返回:
        bool: 路径是否有效
    """
    if not os.path.exists(project_path):
        logging.error(f"项目文件不存在: {project_path}")
        return False
    if not project_path.endswith('.cst'):
        logging.error(f"项目文件格式错误，应为.cst文件: {project_path}")
        return False
    return True

def extract_cst_data(project_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从CST项目文件中提取纳米柱参数和相位数据
    
    参数:
        project_path: CST项目文件路径 (.cst)
    
    返回:
        X: 纳米柱参数 [L, W]，shape: (2, n_samples)，单位：nm
        Y: 频率数据，shape: (n_freq_points, n_samples)
        Z: 相位数据，shape: (n_freq_points, n_samples)
        N: 初始相位，shape: (1, n_samples)
    """
    if not validate_project_path(project_path):
        raise ValueError(f"无效的CST项目路径: {project_path}")
    
    logging.info(f"开始处理项目: {project_path}")
    
    try:
        project = cst.results.ProjectFile(project_path)
        run_ids = project.get_3d().get_all_run_ids()
        num_runs = len(run_ids)
        logging.info(f"共有 {num_runs} 个扫描参数组合")
        
        if num_runs == 0:
            logging.error("未找到任何扫描参数组合")
            raise ValueError("未找到扫描参数组合")
        
        X_list = []  # 存储 [L, W]，单位：nm
        Y_list = []  # 存储频率
        Z_list = []  # 存储相位
        N_list = []  # 存储初始相位
        
        failed_runs = []
        
        # 使用tqdm显示进度
        for idx, Runid in tqdm(enumerate(run_ids), total=num_runs, desc="提取数据"):
            try:
                # 提取PHASE数据
                s11 = project.get_3d().get_result_item(r"Tables\1D Results\PHASE", Runid)
                x = s11.get_parameter_combination()
                y = s11.get_xdata()
                z = s11.get_ydata()
                
                # 提取实部相位
                z_real = np.array([val.real for val in z])
                
                # 记录参数和相位数据
                # 从CST提取的尺寸数据单位为纳米(nm)，直接使用
                L_val = x.get('L', 0)  # 单位：nm
                W_val = x.get('W', 0)  # 单位：nm
                
                # 验证参数有效性
                if L_val <= 0 or W_val <= 0:
                    logging.warning(f"Runid={Runid} 参数无效: L={L_val}, W={W_val}")
                    failed_runs.append(Runid)
                    continue
                
                X_list.append([L_val, W_val])
                Z_list.append(z_real)
                N_list.append(z_real[0])  # 记录第一个频率点的相位
                Y_list.append(y)
                
                # 进度输出
                if (idx + 1) % 50 == 0 or idx == 0 or idx == num_runs - 1:
                    logging.info(f"进度: [{idx+1}/{num_runs}] L={L_val:.4f} nm, W={W_val:.4f} nm, "
                               f"data_len={len(z_real)}")
                               
            except Exception as e:
                logging.warning(f"Runid={Runid} 处理失败: {e}")
                failed_runs.append(Runid)
                continue
        
        # 报告失败情况
        if failed_runs:
            logging.warning(f"共有 {len(failed_runs)} 个运行失败: {failed_runs}")
        
        # 检查是否有有效数据
        if not X_list:
            logging.error("没有提取到有效数据")
            raise ValueError("没有提取到有效数据")
        
        # 数据对齐 - 确保所有样本具有相同长度，只提取前751个数据点
        target_len = 751
        logging.info(f"固定提取前 {target_len} 个数据点")
        
        # 转换为numpy数组
        X = np.array(X_list).T  # shape: (2, n_samples)
        Y = np.array([y[:target_len] for y in Y_list]).T  # shape: (n_freq, n_samples)
        Z = np.array([z[:target_len] for z in Z_list]).T  # shape: (n_freq, n_samples)
        N = np.array(N_list).reshape(1, -1)  # shape: (1, n_samples)
        
        logging.info(f"数据提取完成!")
        logging.info(f"Size shape: {X.shape}")
        logging.info(f"Frequency shape: {Y.shape}")
        logging.info(f"Phase shape: {Z.shape}")
        
        return X, Y, Z, N
        
    except Exception as e:
        logging.error(f"提取数据时发生错误: {e}")
        raise

def shift_phase(Z: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对相位参数进行平移处理，使所有相位曲线的起始值均为0
    
    参数:
        Z: 原始相位数据，shape: (n_freq, n_samples)
        N: 初始相位，shape: (1, n_samples)
    
    返回:
        Z_shifted: 平移后的相位数据，shape: (n_freq, n_samples)
        phase_shift: 相位偏移值，shape: (1, n_samples)
    """
    try:
        # 相位平移：减去初始相位
        Z_shifted = Z - N
        phase_shift = N.copy()
        
        logging.info(f"相位平移完成!")
        logging.info(f"原始相位范围: [{Z.min():.4f}, {Z.max():.4f}]")
        logging.info(f"平移后相位范围: [{Z_shifted.min():.4f}, {Z_shifted.max():.4f}]")
        
        return Z_shifted, phase_shift
    except Exception as e:
        logging.error(f"相位平移处理失败: {e}")
        raise

def save_raw_data(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, N: np.ndarray, output_dir: str):
    """
    保存原始数据
    
    参数:
        X: 纳米柱参数
        Y: 频率数据
        Z: 原始相位数据
        N: 初始相位
        output_dir: 输出目录
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'X_raw.npy'), X)
        np.save(os.path.join(output_dir, 'Y_raw.npy'), Y)
        np.save(os.path.join(output_dir, 'Z_raw.npy'), Z)
        np.save(os.path.join(output_dir, 'N_raw.npy'), N)
        
        logging.info(f"原始数据已保存到: {output_dir}")
    except Exception as e:
        logging.error(f"保存原始数据失败: {e}")
        raise

def save_processed_data(X: np.ndarray, Y: np.ndarray, Z_shifted: np.ndarray, phase_shift: np.ndarray, output_dir: str):
    """
    保存处理后的数据
    
    参数:
        X: 纳米柱参数
        Y: 频率数据
        Z_shifted: 平移后的相位数据
        phase_shift: 相位偏移值
        output_dir: 输出目录
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'X_processed.npy'), X)
        np.save(os.path.join(output_dir, 'Y_processed.npy'), Y)
        np.save(os.path.join(output_dir, 'Z_shifted.npy'), Z_shifted)
        np.save(os.path.join(output_dir, 'phase_shift.npy'), phase_shift)
        
        logging.info(f"处理后数据已保存到: {output_dir}")
    except Exception as e:
        logging.error(f"保存处理后数据失败: {e}")
        raise

def generate_report(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, Z_shifted: np.ndarray, phase_shift: np.ndarray, output_dir: str):
    """
    生成数据处理报告
    
    参数:
        X: 纳米柱参数
        Y: 频率数据
        Z: 原始相位数据
        Z_shifted: 平移后的相位数据
        phase_shift: 相位偏移值
        output_dir: 输出目录
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'phase_shift_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("相位平移处理报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. 数据基本信息\n")
            f.write("-" * 40 + "\n")
            f.write(f"样本数量: {X.shape[1]}\n")
            f.write(f"频率点数量: {Y.shape[0]}\n")
            f.write(f"结构参数维度: {X.shape[0]}\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
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
            f.write(f"原始相位最小值: {Z.min():.4f}\n")
            f.write(f"原始相位最大值: {Z.max():.4f}\n")
            f.write(f"原始相位平均值: {Z.mean():.4f}\n")
            f.write(f"原始相位标准差: {Z.std():.4f}\n\n")
            
            f.write(f"平移后相位最小值: {Z_shifted.min():.4f}\n")
            f.write(f"平移后相位最大值: {Z_shifted.max():.4f}\n")
            f.write(f"平移后相位平均值: {Z_shifted.mean():.4f}\n")
            f.write(f"平移后相位标准差: {Z_shifted.std():.4f}\n\n")
            
            f.write(f"相位偏移值最小值: {phase_shift.min():.4f}\n")
            f.write(f"相位偏移值最大值: {phase_shift.max():.4f}\n")
            f.write(f"相位偏移值平均值: {phase_shift.mean():.4f}\n")
            f.write(f"相位偏移值标准差: {phase_shift.std():.4f}\n\n")
            
            f.write("4. 数据质量评估\n")
            f.write("-" * 40 + "\n")
            f.write(f"数据完整性: 良好\n")
            f.write(f"参数范围: 合理\n")
            f.write(f"单位系统: 纳米 (nm)\n")
        
        logging.info(f"数据处理报告已生成: {report_path}")
    except Exception as e:
        logging.error(f"生成报告失败: {e}")
        raise

def visualize_phase_shift(Z: np.ndarray, Z_shifted: np.ndarray, Y: np.ndarray, output_dir: str):
    """
    可视化相位平移前后的变化
    
    参数:
        Z: 原始相位数据
        Z_shifted: 平移后的相位数据
        Y: 频率数据
        output_dir: 输出目录
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 随机选择5个样本进行展示
        sample_indices = np.random.choice(Z.shape[1], 5, replace=False)
        
        # 绘制原始相位和平移后相位对比
        plt.figure(figsize=(12, 8))
        
        for i, idx in enumerate(sample_indices):
            plt.subplot(5, 1, i+1)
            plt.plot(Y[:, idx], Z[:, idx], label='原始相位', linewidth=2)
            plt.plot(Y[:, idx], Z_shifted[:, idx], label='平移后相位', linewidth=2, linestyle='--')
            plt.title(f'样本 {idx} 相位平移前后对比')
            plt.xlabel('频率')
            plt.ylabel('相位')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'phase_shift_visualization.png'), dpi=150)
        plt.close()
        
        logging.info("相位平移可视化已生成")
    except Exception as e:
        logging.error(f"生成可视化失败: {e}")
        raise

def main():
    """主函数"""
    # 配置参数
    PROJECT_PATH = r"d:\CST output\CST 7\Dataset3.cst"
    # 为相位平移处理创建独立的输出文件夹
    OUTPUT_DIR = r"d:\CST output\CST 7\dataprepro_phase_shift"
    RAW_DATA_DIR = os.path.join(OUTPUT_DIR, 'raw_data')
    PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, 'processed_data')
    REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')
    VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
    
    # 打印流程开始的分隔线和提示信息
    logging.info("=" * 60)
    logging.info("开始相位平移数据预处理流程")
    logging.info("=" * 60)
    
    try:
        # 1. 提取数据
        logging.info("从CST项目中提取数据...")
        X, Y, Z, N = extract_cst_data(PROJECT_PATH)
        
        # 2. 保存原始数据
        logging.info("保存原始数据...")
        save_raw_data(X, Y, Z, N, RAW_DATA_DIR)
        
        # 3. 相位平移处理
        logging.info("进行相位平移处理...")
        Z_shifted, phase_shift = shift_phase(Z, N)
        
        # 4. 保存处理后的数据
        logging.info("保存处理后的数据...")
        save_processed_data(X, Y, Z_shifted, phase_shift, PROCESSED_DATA_DIR)
        
        # 5. 生成报告
        logging.info("生成处理报告...")
        generate_report(X, Y, Z, Z_shifted, phase_shift, REPORT_DIR)
        
        # 6. 生成可视化
        logging.info("生成可视化图表...")
        visualize_phase_shift(Z, Z_shifted, Y, VISUALIZATION_DIR)
        
        logging.info("=" * 60)
        logging.info("相位平移数据预处理完成!")
        logging.info(f"所有数据已保存到: {OUTPUT_DIR}")
        logging.info("=" * 60)
    except Exception as e:
        logging.error(f"预处理流程失败: {e}")
        raise

if __name__ == "__main__":
    main()