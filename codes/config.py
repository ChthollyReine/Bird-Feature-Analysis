"""全局配置：路径、随机种子、算法参数。

可复现性要求：所有随机性（抽样、聚类初始化、PCA 无随机性）统一由 SEED 控制。
"""
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent

# 数据
DATA_ZIP = ROOT / "data" / "archive.zip"
SAMPLE_ZIP = ROOT / "data" / "sample_images.zip"  # 精简版：每类 1 张，供云端展示缩略图
CUB_PREFIX = "CUB_200_2011"

# 输出目录
OUTPUT_DIR = ROOT / "output"

# 随机种子（可复现性）
SEED = 42

# 骨架阶段运行规模：抽样 3 个类别、每类 10 张，先跑通流程
N_CLASSES = 3
PER_CLASS = 10

# 预处理参数
PREPROCESS = {
    "bilateral_d": 9,
    "bilateral_sigma_color": 75,
    "bilateral_sigma_space": 75,
    "clahe_clip": 2.0,
    "clahe_grid": (8, 8),
}

# 分割参数
SEGMENT = {
    "blur_ksize": 5,
    "morph_open_ksize": 3,
}

# 灰度共生矩阵（GLCM）参数
GLCM = {
    "levels": 32,
    "distances": [1, 2],
    "angles": [0, 45, 90, 135],
}

# 分形维数（Box-counting）参数
FRACTAL = {
    "box_sizes": [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64],
    "threshold": 128,
}

# 颜色特征参数
COLOR = {
    "h_bins": 8,
    "s_bins": 8,
    "v_bins": 8,
}

# PCA / 聚类参数
ANALYSIS = {
    "n_components": 2,
    "n_clusters": 3,      # KMeans 聚类数；设为 None 则跳过聚类
}
