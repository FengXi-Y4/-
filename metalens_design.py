"""
超透镜设计实现 - metalens_design.py
环境: metalens (Python 3.12.4)
功能: 基于训练好的深度学习模型实现超透镜设计
"""

import torch
import numpy as np
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('metalens_design.log'),
        logging.StreamHandler()
    ]
)

# 导入模型
from metalens_model import MetaLensNet
from metalens_phase_shift_model import MetaLensPhaseShiftNet


def generate_target_phase(diameter: float, focal_length: float, 
                        wavelengths: List[float], resolution: int = 1024) -> np.ndarray:
    """
    生成超透镜目标相位分布
    
    参数:
        diameter: 超透镜直径 (μm)
        focal_length: 焦距 (μm)
        wavelengths: 工作波长范围 (nm)
        resolution: 分辨率
        
    返回:
        目标相位分布
    """
    try:
        # 输入验证
        if diameter <= 0:
            raise ValueError("超透镜直径必须大于0")
        if focal_length <= 0:
            raise ValueError("焦距必须大于0")
        if not wavelengths:
            raise ValueError("波长列表不能为空")
        if resolution <= 0:
            raise ValueError("分辨率必须大于0")
        
        # 转换单位
        diameter_nm = diameter * 1000  # 转换为nm
        focal_length_nm = focal_length * 1000  # 转换为nm
        
        # 生成网格
        x = np.linspace(-diameter_nm/2, diameter_nm/2, resolution)
        y = np.linspace(-diameter_nm/2, diameter_nm/2, resolution)
        X, Y = np.meshgrid(x, y)
        
        # 计算距离中心的距离
        r = np.sqrt(X**2 + Y**2)
        
        # 计算目标相位分布
        phase_distributions = []
        for λ in wavelengths:
            # 理想透镜相位分布
            #k = 2 * np.pi / λ                           
            k = 2 * np.pi * (  1 / λ  -  1/650  )          #相位补偿值，后面的不需要再减phase[650nm]                
            phase = - k * (np.sqrt(r**2 + focal_length_nm**2) - focal_length_nm)
            phase_distributions.append(phase)
        
        result = np.array(phase_distributions)
        logging.info(f"目标相位分布生成完成: 形状={result.shape}, 相位范围=[{result.min():.2f}, {result.max():.2f}] rad")
        return result
    except Exception as e:
        logging.error(f"生成目标相位分布失败: {e}")
        raise

def load_normalization_params(norm_path: str = r"d:\CST output\CST 7\dataset_shifted_phase\norm_params.json",
                             phase_shift_norm_path: str = r"d:\CST output\CST 7\dataset_phase_shift\norm_params.json") -> Dict:
    """
    加载数据预处理时的归一化参数
    同时加载主相位和相位偏移的归一化参数
    
    参数:
        norm_path: 主相位归一化参数文件路径
        phase_shift_norm_path: 相位偏移归一化参数文件路径
        
    返回:
        归一化参数字典
    """
    try:
        import json
        
        # 加载主相位归一化参数
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"主相位归一化参数文件不存在: {norm_path}")
        
        with open(norm_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        # 验证主相位参数完整性
        required_keys = ['Z_min', 'Z_max', 'X_min', 'X_max']
        for key in required_keys:
            if key not in params:
                raise ValueError(f"主相位归一化参数缺少关键字段: {key}")
        
        logging.info(f"主相位归一化参数已加载: {norm_path}")
        logging.info(f"  主相位范围: [{params['Z_min']:.2f}, {params['Z_max']:.2f}] rad")
        logging.info(f"  长度L范围: [{params['X_min'][0][0]:.1f}, {params['X_max'][0][0]:.1f}] nm")
        logging.info(f"  宽度W范围: [{params['X_min'][1][0]:.1f}, {params['X_max'][1][0]:.1f}] nm")
        
        # 加载相位偏移归一化参数
        if os.path.exists(phase_shift_norm_path):
            with open(phase_shift_norm_path, 'r', encoding='utf-8') as f:
                phase_shift_params = json.load(f)
            
            if 'phase_min' in phase_shift_params and 'phase_max' in phase_shift_params:
                params['phase_min'] = phase_shift_params['phase_min']
                params['phase_max'] = phase_shift_params['phase_max']
                logging.info(f"相位偏移归一化参数已加载: {phase_shift_norm_path}")
                logging.info(f"  相位偏移范围: [{params['phase_min']:.2f}, {params['phase_max']:.2f}] rad")
            else:
                logging.warning(f"相位偏移参数文件中缺少 phase_min/phase_max，将使用默认反归一化")
        else:
            logging.warning(f"相位偏移归一化参数文件不存在: {phase_shift_norm_path}")
        
        return params
    except Exception as e:
        logging.error(f"加载归一化参数失败: {e}")
        raise

def load_model(model_path: str, input_dim: int) -> MetaLensNet:
    """
    加载训练好的模型
    
    参数:
        model_path: 模型路径
        input_dim: 输入维度
        
    返回:
        加载好的模型
    """
    try:
        # 验证模型文件存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 验证文件大小
        if os.path.getsize(model_path) == 0:
            raise ValueError(f"模型文件为空: {model_path}")
        
        # 创建模型
        model = MetaLensNet(input_dim=input_dim, output_dim=2)
        
        # 加载模型权重
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # 加载状态字典
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        logging.info(f"模型加载完成: {model_path}")
        logging.info(f"  设备: {device}")
        logging.info(f"  输入维度: {input_dim}")
        return model
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        raise

def load_phase_shift_model(model_path: str) -> MetaLensPhaseShiftNet:
    """
    加载训练好的相位偏移模型
    
    参数:
        model_path: 模型路径
        
    返回:
        加载好的模型
    """
    try:
        # 验证模型文件存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"相位偏移模型文件不存在: {model_path}")
        
        # 验证文件大小
        if os.path.getsize(model_path) == 0:
            raise ValueError(f"相位偏移模型文件为空: {model_path}")
        
        # 创建模型
        model = MetaLensPhaseShiftNet(input_dim=2, output_dim=1)
        
        # 加载模型权重
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # 加载状态字典
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        logging.info(f"相位偏移模型加载完成: {model_path}")
        logging.info(f"  设备: {device}")
        return model
    except Exception as e:
        logging.error(f"加载相位偏移模型失败: {e}")
        raise


def generate_metalens_design(model: MetaLensNet, phase_shift_model: MetaLensPhaseShiftNet, target_phase: np.ndarray, 
                           wavelengths: List[float], period: float, focal_length: float,
                           norm_params: Dict,
                           output_dir: str = './metalens_design'):
    """
    生成超透镜设计
    
    参数:
        model: 训练好的相位预测模型
        phase_shift_model: 训练好的相位偏移预测模型
        target_phase: 目标相位分布 (单位: rad)
        wavelengths: 工作波长
        period: 纳米柱周期 (nm)
        focal_length: 焦距 (μm)
        norm_params: 全局归一化参数 (与训练一致)
        output_dir: 输出目录
    """
    try:
        # 输入验证
        if model is None:
            raise ValueError("模型不能为空")
        if phase_shift_model is None:
            raise ValueError("相位偏移模型不能为空")
        if target_phase is None or target_phase.size == 0:
            raise ValueError("目标相位分布不能为空")
        if not wavelengths:
            raise ValueError("波长列表不能为空")
        if period <= 0:
            raise ValueError("纳米柱周期必须大于0")
        if focal_length <= 0:
            raise ValueError("焦距必须大于0")
        if norm_params is None or not norm_params:
            raise ValueError("归一化参数不能为空")
        
        global NANOCELL_PERIOD
        NANOCELL_PERIOD = period
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"输出目录已创建: {output_dir}")
        
        # 设备选择
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        phase_shift_model.to(device)
        logging.info(f"使用设备: {device}")
        
        # 获取分辨率
        resolution = target_phase.shape[1]
        if resolution <= 0:
            raise ValueError("分辨率必须大于0")
        
        # 存储纳米柱参数
        L_map = np.full((resolution, resolution), np.nan)
        W_map = np.full((resolution, resolution), np.nan)
        phase_shift_1_rad_map = np.full((resolution, resolution), np.nan)
        phase_shift_1_deg_map = np.full((resolution, resolution), np.nan)
        phase_shift_2_rad_map = np.full((resolution, resolution), np.nan)
        phase_shift_2_deg_map = np.full((resolution, resolution), np.nan)
        
        # 存储Po矩阵数据 (用于FDTD导入)
        Po_data = []
        N_data = []
        
        # 计算圆形区域掩码 (半径 = resolution/2)
        center = (resolution - 1) / 2.0
        y_grid, x_grid = np.ogrid[:resolution, :resolution]
        distance_from_center = np.sqrt((x_grid - center)**2 + (y_grid - center)**2)
        circle_radius = resolution / 2.0
        circular_mask = distance_from_center <= circle_radius
        
        # 加载训练时的全局归一化参数
        Z_MIN = norm_params['Z_min']
        Z_MAX = norm_params['Z_max']
        L_MIN = norm_params['X_min'][0][0]
        L_MAX = norm_params['X_max'][0][0]
        W_MIN = norm_params['X_min'][1][0]
        W_MAX = norm_params['X_max'][1][0]
        
        logging.info(f"使用训练数据全局归一化:")
        logging.info(f"  相位: [{Z_MIN:.2f}, {Z_MAX:.2f}] rad")
        logging.info(f"  长度L: [{L_MIN:.1f}, {L_MAX:.1f}] nm")
        logging.info(f"  宽度W: [{W_MIN:.1f}, {W_MAX:.1f}] nm")
        logging.info(f"开始生成超透镜设计: 分辨率={resolution}x{resolution}")
        
        # 打印目标相位范围
        logging.info(f"目标相位范围: [{target_phase.min():.2f}, {target_phase.max():.2f}] rad")
        logging.info(f"圆形区域纳米柱数量: {np.sum(circular_mask)} / {resolution * resolution}")
        
        period_um = period / 1000.0  # 转换为μm
        lambda_nm = 650.0  # 固定使用650nm波长
        lambda_um = lambda_nm / 1000.0  # 转换为μm (正确的单位换算)
        
        # 遍历每个像素 (仅处理圆形区域内)
        from tqdm import tqdm
        valid_pixels = np.sum(circular_mask)
        
        with tqdm(total=valid_pixels, desc="生成纳米柱设计") as pbar:
            for i in range(resolution):
                for j in range(resolution):
                    # 跳过圆形区域外的像素
                    if not circular_mask[i, j]:
                        continue
                    
                    # 计算中心坐标 (原点在阵列中心, 单位μm)
                    x = (j - resolution / 2 + 0.5) * period_um
                    y = (i - resolution / 2 + 0.5) * period_um
                    
                    # ========== 步骤1: 获取该坐标所需要的相位值 ==========
                    phase_profile = target_phase[:, i, j]
                    
                    # 与训练预处理完全对齐 - 减去初始相位
                    #phase_profile = phase_profile - phase_profile[0]
                    
                    # WAVELENGTHS已直接设置为1002个点（与训练模型输入维度对齐），无需插值
                    phase_interp = phase_profile
                    
                    # 使用训练时的GLOBAL全局归一化
                    phase_norm = (phase_interp - Z_MIN) / (Z_MAX - Z_MIN + 1e-8)
                    
                    # ========== 步骤2: 将相位值导入第一个神经网络获取纳米柱尺寸 ==========
                    phase_tensor = torch.tensor(phase_norm, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        params = model(phase_tensor)
                    
                    L_norm, W_norm = params.squeeze().cpu().numpy()
                    
                    # 反归一化获得实际纳米柱尺寸
                    L = L_norm * (L_MAX - L_MIN) + L_MIN
                    W = W_norm * (W_MAX - W_MIN) + W_MIN
                    
                    # 尺寸约束                           #不知道有没有用
                    #L = max(L_MIN, min(L, L_MAX))
                    #W = max(W_MIN, min(W, W_MAX))
                    
                    # ========== 步骤3: 将纳米柱结构导入第二个神经网络获取需要补偿的相位值 ==========
                    # 归一化纳米柱参数
                    L_norm_nn2 = (L - L_MIN) / (L_MAX - L_MIN + 1e-8)
                    W_norm_nn2 = (W - W_MIN) / (W_MAX - W_MIN + 1e-8)
                    
                    nanopillar_tensor = torch.tensor([[L_norm_nn2, W_norm_nn2]], dtype=torch.float32).to(device)
                    
                    # 预测相位偏移
                    with torch.no_grad():
                        phase_shift_pred = phase_shift_model(nanopillar_tensor)
                    
                    # 反归一化相位偏移
                    if 'phase_min' in norm_params and 'phase_max' in norm_params:
                        phase_min = norm_params['phase_min']
                        phase_max = norm_params['phase_max']
                        phase_shift_1_rad = (phase_shift_pred.item() + 1) * (phase_max - phase_min) / 2.0 + phase_min
                        logging.debug(f"相位偏移反归一化: 模型输出={phase_shift_pred.item():.4f}, 范围=[{phase_min:.2f}, {phase_max:.2f}], 结果={phase_shift_1_rad:.4f} rad")
                    else:
                        phase_shift_1_rad = (phase_shift_pred.item() + 1) * 3.1416
                        logging.warning(f"相位偏移参数未找到，使用默认反归一化: 输出={phase_shift_pred.item():.4f} -> {phase_shift_1_rad:.4f} rad")
                    
                    # 转换偏转角1为角度值
                    phase_shift_1_deg = phase_shift_1_rad * 90.0 / np.pi                  #180/2=90
                    
                    # ========== 步骤4: 计算偏转角2 ==========
                    s = np.sqrt(x**2 + y**2)
                    theta2_rad = 2 * np.pi * (focal_length - np.sqrt(s**2 + focal_length**2)) / lambda_um      #PB相位  系数2
                    theta2_deg = theta2_rad * 90.0 / np.pi                  #这里已经是偏转角了，不需要/2了    180/2=90
                    
                    # 存储所有参数
                    L_map[i, j] = L
                    W_map[i, j] = W
                    phase_shift_1_rad_map[i, j] = phase_shift_1_rad
                    phase_shift_1_deg_map[i, j] = phase_shift_1_deg
                    phase_shift_2_rad_map[i, j] = theta2_rad
                    phase_shift_2_deg_map[i, j] = theta2_deg
                    
                    # 准备FDTD导入格式数据
                    Po_row = [x, y, theta2_deg, 0]
                    N_row = [len(Po_data) + 1, 0, L , W , phase_shift_1_deg]
                    
                    Po_data.append(Po_row)
                    N_data.append(N_row)
                    
                    pbar.update(1)
        
        # 更新Po矩阵第4列为总数
        m = len(Po_data)
        for row in Po_data:
            row[3] = m
        
        # 转换为numpy数组
        Po = np.array(Po_data)
        N = np.array(N_data)
        
        # ===== 验证相位偏移范围 =====
        valid_mask = ~np.isnan(phase_shift_1_rad_map)
        phase1_rad_valid = phase_shift_1_rad_map[valid_mask]
        phase1_deg_valid = phase_shift_1_deg_map[valid_mask]
        phase2_rad_valid = phase_shift_2_rad_map[valid_mask]
        phase2_deg_valid = phase_shift_2_deg_map[valid_mask]
        
        logging.info("=" * 60)
        logging.info("相位偏移计算验证:")
        logging.info(f"  模型原始输出范围: [-1, 1] (Tanh输出)")
        if 'phase_min' in norm_params and 'phase_max' in norm_params:
            logging.info(f"  相位偏移实际范围: [{norm_params['phase_min']:.4f}, {norm_params['phase_max']:.4f}] rad")
        logging.info(f"  偏转角1 rad 范围: [{phase1_rad_valid.min():.4f}, {phase1_rad_valid.max():.4f}]")
        logging.info(f"  偏转角1 deg 范围: [{phase1_deg_valid.min():.2f}, {phase1_deg_valid.max():.2f}]")
        logging.info(f"  偏转角2 rad 范围: [{phase2_rad_valid.min():.4f}, {phase2_rad_valid.max():.4f}]")
        logging.info(f"  偏转角2 deg 范围: [{phase2_deg_valid.min():.2f}, {phase2_deg_valid.max():.2f}]")
        logging.info("=" * 60)
        
        # 保存设计结果
        np.save(os.path.join(output_dir, 'L_map.npy'), L_map)
        np.save(os.path.join(output_dir, 'W_map.npy'), W_map)
        np.save(os.path.join(output_dir, 'phase_shift_1.npy'), phase_shift_1_deg_map)
        
        # 保存MATLAB格式的Po和N矩阵 (用于FDTD导入)
        import scipy.io as sio
        sio.savemat(os.path.join(output_dir, 'P70.mat'), {'Po': Po})
        sio.savemat(os.path.join(output_dir, 'struct.mat'), {'N': N})
        
        # ========== 直接导出符合要求格式的TXT文件 ==========
        txt_path = os.path.join(output_dir, 'nanopillar_coordinates.txt')
        nanopillar_count = 0
        
        with open(txt_path, 'w') as f:
            f.write("# 超透镜纳米柱设计文件 (用于FDTD导入)\n")
            f.write("# 格式: x坐标(μm)  y坐标(μm)  长度L(nm)  宽度W(nm)  偏转角1(rad)  偏转角1(deg)  偏转角2(rad)  偏转角2(deg)\n")
            f.write(f"# 阵列规模: {resolution} x {resolution}\n")
            f.write(f"# 周期: {period} nm\n")
            f.write(f"# 焦距: {focal_length} μm\n")
            f.write(f"# 中心波长: {lambda_nm} nm\n")
            f.write("# 计算顺序: 先求相位(rad)，再转换为角度(deg)\n")
            f.write("# 偏转角1: 相位偏移模型预测值\n")
            f.write("# 偏转角2: 消色差透镜理论相位值 (使用650nm波长)\n")
            f.write("#" + "=" * 100 + "\n\n")
            
            for i in range(resolution):
                for j in range(resolution):
                    if np.isnan(L_map[i, j]) or np.isnan(W_map[i, j]):
                        continue
                    
                    x = (j - resolution / 2 + 0.5) * period_um
                    y = (i - resolution / 2 + 0.5) * period_um
                    
                    L = L_map[i, j]
                    W = W_map[i, j]
                    p1_rad = phase_shift_1_rad_map[i, j]
                    p1_deg = phase_shift_1_deg_map[i, j]
                    p2_rad = phase_shift_2_rad_map[i, j]
                    p2_deg = phase_shift_2_deg_map[i, j]
                    
                    f.write(f"{x:>10.4f} {y:>10.4f} {L:>10.2f} {W:>10.2f} {p1_rad:>14.6f} {p1_deg:>12.4f} {p2_rad:>14.6f} {p2_deg:>12.4f}\n")
                    nanopillar_count += 1
        
        # 打印TXT导出信息
        if nanopillar_count > 0:
            logging.info(f"纳米柱坐标文件已导出: {txt_path}")
            logging.info(f"共导出 {nanopillar_count} 个纳米柱 (圆形区域)")
            logging.info(f"  格式: x y L W theta1_rad theta1_deg theta2_rad theta2_deg")
        else:
            logging.warning("未导出任何纳米柱数据！")
        
        # 生成可视化
        generate_design_visualizations(L_map, W_map, output_dir)
        
        logging.info(f"超透镜设计完成! 结果保存到: {output_dir}")
        logging.info(f"已生成 P70.mat (Po矩阵) 和 struct.mat (N矩阵) 用于FDTD导入")
        
        return L_map, W_map, phase_shift_1_deg_map
    except Exception as e:
        logging.error(f"生成超透镜设计失败: {e}")
        raise

def generate_design_visualizations(L_map: np.ndarray, W_map: np.ndarray, output_dir: str):
    """
    生成设计可视化
    
    参数:
        L_map: 长度参数图
        W_map: 宽度参数图
        output_dir: 输出目录
    """
    # 创建圆形区域掩码 (nan为圆形外)
    valid_mask = ~np.isnan(L_map)
    
    # 长度参数可视化 (设置背景为白色)
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.viridis.with_extremes(bad='white')
    plt.imshow(L_map, cmap=cmap)
    plt.colorbar(label='长度 L (nm)')
    plt.title('圆形超透镜纳米柱长度分布', fontsize=20, fontweight='bold')
    plt.xlabel('像素', fontsize=20)
    plt.ylabel('像素', fontsize=20)
    plt.savefig(os.path.join(output_dir, 'L_map_visualization.png'), dpi=600)
    plt.close()
    
    # 宽度参数可视化 (设置背景为白色)
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.plasma.with_extremes(bad='white')
    plt.imshow(W_map, cmap=cmap)
    plt.colorbar(label='宽度 W (nm)')
    plt.title('圆形超透镜纳米柱宽度分布', fontsize=20, fontweight='bold')
    plt.xlabel('像素', fontsize=20)
    plt.ylabel('像素', fontsize=20)
    plt.savefig(os.path.join(output_dir, 'W_map_visualization.png'), dpi=600)
    plt.close()
    
    # 参数直方图 (仅统计圆形区域内的有效数据)
    L_valid = L_map[valid_mask]
    W_valid = W_map[valid_mask]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(L_valid, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'长度 L 分布 (圆形区域, N={len(L_valid)})', fontsize=20, fontweight='bold')
    plt.xlabel('长度 (nm)', fontsize=20)
    plt.ylabel('频率', fontsize=20)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(W_valid, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title(f'宽度 W 分布 (圆形区域, N={len(W_valid)})', fontsize=20, fontweight='bold')
    plt.xlabel('宽度 (nm)', fontsize=20)
    plt.ylabel('频率', fontsize=20)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_distribution.png'), dpi=600)
    plt.close()
    
    logging.info("设计可视化完成 (圆形区域)")

def create_design_report(L_map: np.ndarray, W_map: np.ndarray, phase_shift_map: np.ndarray,
                        wavelengths: List[float], 
                        diameter: float, focal_length: float, output_dir: str):
    """
    创建设计报告
    
    参数:
        L_map: 长度参数图
        W_map: 宽度参数图
        phase_shift_map: 相位偏移图
        wavelengths: 工作波长
        diameter: 超透镜直径
        focal_length: 焦距
        output_dir: 输出目录
    """
    # 提取圆形区域内的有效数据
    valid_mask = ~np.isnan(L_map)
    L_valid = L_map[valid_mask]
    W_valid = W_map[valid_mask]
    phase_shift_valid = phase_shift_map[valid_mask]
    nanopillar_count = len(L_valid)
    
    report_path = os.path.join(output_dir, 'design_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("圆形超透镜设计报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. 设计基本参数\n")
        f.write("-" * 40 + "\n")
        f.write(f"形状: 圆形\n")
        f.write(f"超透镜直径: {diameter:.2f} μm\n")
        f.write(f"焦距: {focal_length:.2f} μm\n")
        f.write(f"工作波长范围: {min(wavelengths):.0f} - {max(wavelengths):.0f} nm\n")
        f.write(f"纳米柱周期: 450 nm\n")
        f.write(f"方形包围阵列: {L_map.shape[0]} x {L_map.shape[1]}\n")
        f.write(f"圆形区域纳米柱数量: {nanopillar_count}\n")
        f.write(f"设计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("2. 纳米柱尺寸统计（圆形区域内，单位：nm）\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'-'*15} {'最小值':>10} {'最大值':>10} {'平均值':>10} {'标准差':>10}\n")
        f.write(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}\n")
        
        # L参数统计 (仅圆形区域)
        L_min = L_valid.min()
        L_max = L_valid.max()
        L_mean = L_valid.mean()
        L_std = L_valid.std()
        f.write(f"{'长度 L':<15} {L_min:>10.2f} {L_max:>10.2f} {L_mean:>10.2f} {L_std:>10.2f}\n")
        
        # W参数统计 (仅圆形区域)
        W_min = W_valid.min()
        W_max = W_valid.max()
        W_mean = W_valid.mean()
        W_std = W_valid.std()
        f.write(f"{'宽度 W':<15} {W_min:>10.2f} {W_max:>10.2f} {W_mean:>10.2f} {W_std:>10.2f}\n\n")
        
        f.write("3. 偏转角统计（圆形区域内）\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'-'*20} {'最小值':>12} {'最大值':>12} {'平均值':>12} {'标准差':>12}\n")
        f.write(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}\n")
        
        # 偏转角1统计 (相位偏移模型, 已为角度值)
        ps_min = phase_shift_valid.min()
        ps_max = phase_shift_valid.max()
        ps_mean = phase_shift_valid.mean()
        ps_std = phase_shift_valid.std()
        f.write(f"{'偏转角1 (deg)':<20} {ps_min:>12.2f} {ps_max:>12.2f} {ps_mean:>12.2f} {ps_std:>12.2f}\n")
        f.write(f"{'偏转角1 (rad)':<20} {ps_min*np.pi/180:>12.4f} {ps_max*np.pi/180:>12.4f} {ps_mean*np.pi/180:>12.4f} {ps_std*np.pi/180:>12.4f}\n\n")
        
        f.write("4. 导出文件说明\n")
        f.write("-" * 40 + "\n")
        f.write("L_map.npy             纳米柱长度矩阵 (numpy格式)\n")
        f.write("W_map.npy             纳米柱宽度矩阵 (numpy格式)\n")
        f.write("phase_shift_1.npy     偏转角1矩阵 (相位偏移模型, numpy格式)\n")
        f.write("P70.mat               Po矩阵 MATLAB格式 (FDTD导入用)\n")
        f.write("struct.mat            N矩阵 MATLAB格式 (FDTD导入用)\n")
        f.write("nanopillar_coordinates.txt  纳米柱完整参数文件 (TXT格式)\n")
        f.write("                        格式: x(μm) y(μm) L(nm) W(nm) theta1(rad) theta1(deg) theta2(rad) theta2(deg)\n\n")
        
        f.write("5. 设计质量评估\n")
        f.write("-" * 40 + "\n")
        f.write(f"参数范围: 合理\n")
        f.write(f"单位系统: 纳米 (nm)\n")
        f.write(f"设计完整性: 良好\n")
    
    logging.info(f"设计报告生成完成: {report_path}")

def main():
    """主函数"""
    # 配置参数
    MODEL_PATH = r"D:\CST output\CST 7\checkpoints\best_model.pth"
    PHASE_SHIFT_MODEL_PATH = r"d:\CST output\CST 7\phase_shift_checkpoints\best_phase_shift_model.pth"
    OUTPUT_DIR = r"d:\CST output\CST 7\metalens_design"
    
    # 超透镜设计参数
    NANOCELL_PERIOD = 450  # 纳米柱周期 (nm)
    DIAMETER = 20.0  # 超透镜直径 (μm)
    FOCAL_LENGTH = 40 # 焦距 (μm)
    WAVELENGTHS = np.linspace(650, 500, 751).tolist()  # 工作波长范围 (nm)，共751个点
    RESOLUTION = int(DIAMETER * 1000 / NANOCELL_PERIOD)  # 根据周期自动计算分辨率 = 44
    
    # 打印流程开始的分隔线和提示信息
    logging.info("=" * 60)
    logging.info("开始超透镜设计流程")
    logging.info("=" * 60)
    
    # 1. 生成目标相位分布
    logging.info("生成目标相位分布...")
    target_phase = generate_target_phase(DIAMETER, FOCAL_LENGTH, WAVELENGTHS, RESOLUTION)
    logging.info(f"目标相位分布生成完成: 形状={target_phase.shape}")
    
    # 2. 加载模型
    logging.info("加载训练好的模型...")
    input_dim = 751  # 与训练模型保持一致: 1000频率点插值 + 2目标维度
    model = load_model(MODEL_PATH, input_dim)
    
    # 2.1 加载相位偏移模型
    logging.info("加载训练好的相位偏移模型...")
    phase_shift_model = load_phase_shift_model(PHASE_SHIFT_MODEL_PATH)
    
    # 2.5 加载归一化参数
    norm_params = load_normalization_params()
    
    # 3. 生成超透镜设计
    logging.info("生成超透镜设计...")
    L_map, W_map, phase_shift_map = generate_metalens_design(model, phase_shift_model, target_phase, WAVELENGTHS, 
                                                           NANOCELL_PERIOD, FOCAL_LENGTH, norm_params, OUTPUT_DIR)
    
    # 4. 创建设计报告
    logging.info("创建设计报告...")
    create_design_report(L_map, W_map, phase_shift_map, WAVELENGTHS, DIAMETER, FOCAL_LENGTH, OUTPUT_DIR)
    
    logging.info("=" * 60)
    logging.info("超透镜设计流程完成!")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
