# 鸟类图案生态分析（CUB-200-2011 生态图像特征工程）

基于经典数字图像处理（DIP）的鸟类体表图案「对称性 / 复杂度」量化分析系统：提取羽毛图案特征，并与 AVONET 生态性状关联，探讨鸟类对环境适应性的形态机制。

## 项目简介

- **生态学背景**：量化动物体表图案（如鸟类羽毛纹理、斑点）的对称性与复杂度，研究鸟类对环境的适应性。
- **DIP 核心算法**：图像对齐（仿射/透视）、灰度共生矩阵（GLCM）、分形维数（Box-counting）、对称轴检测与双边纹理相关。
- **AI 辅助**：PCA 主成分分析 + 空间聚类图谱。
- **严格约束**：核心分割与特征提取只用 OpenCV 基础算子组合实现，**不使用 SAM / YOLOv8**。

## 主要功能

1. **完整流水线**：输入 → 去噪滤波 → 几何增强 → 分割 → 特征提取（形貌/纹理/颜色/对称性）→ PCA/聚类。
2. **物种识别**：标准化欧氏距离最近邻，把上传图像匹配到 200 种鸟类并输出其生态指标（轻量分类器，非大模型）。
3. **生态指标映射**：200 种鸟类 ↔ AVONET 性状（体重、喙形、栖息地、迁徙、食性、分布范围等），并做相关性/机理解释。
4. **图形界面**（Streamlit）：浏览/检索、数据集统计、图像分析识别、生态机理解释。

## 目录结构

```
CV/
├─ app.py                      # Streamlit 图形界面入口
├─ codes/
│   ├─ config.py               # 全局配置与随机种子
│   ├─ data_loader.py          # 从 archive.zip 读取 CUB 数据
│   ├─ preprocessing.py        # 去噪/增强/对齐
│   ├─ segmentation.py         # 传统分割（Otsu + 形态学 + 最大连通域）
│   ├─ analysis.py             # PCA + KMeans
│   ├─ ecological.py           # 学名/中文名映射 + AVONET 读取
│   ├─ ecological_analysis.py  # 特征聚合 + 物种识别 + 相关性分析
│   ├─ main.py                 # 流水线骨架（小样本跑通）
│   └─ features/
│       ├─ texture.py          # GLCM 灰度共生矩阵
│       ├─ fractal.py          # Box-counting 分形维数
│       ├─ symmetry.py         # 对称轴检测 + 双边纹理相关
│       └─ color_shape.py      # 颜色直方图 + 轮廓形貌
├─ data/                       # 数据目录（大数据集见 .gitignore，不提交）
│   └─ cub_species_mapping.csv # 200 种鸟类 俗名/学名/中文名 对照
└─ output/                     # 运行生成的产物（特征库、图表）
```

## 环境要求与安装

- Python 3.10+（本项目在 3.13 上验证）
- 依赖：`numpy` `opencv-python` `scikit-learn` `scipy` `pandas` `matplotlib` `openpyxl` `streamlit`

```bash
pip install numpy opencv-python scikit-learn scipy pandas matplotlib openpyxl streamlit
```

## 数据下载（需自行获取，仓库不包含大数据集）

| 文件 | 内容 | 来源 |
|------|------|------|
| `data/archive.zip` | CUB-200-2011 数据集（含 images / parts / attributes） | Caltech-UCSD 官方：https://www.vision.caltech.edu/datasets/cub_200_2011/ ，或 Kaggle 上的 CUB-200-2011 镜像 |
| `data/AVONET Supplementary dataset 1.xlsx` | AVONET 鸟类形态/生态/地理性状表 | Tobias et al. (2022) *AVONET*, Ecology Letters 的补充材料（Figshare） |

> 注意：这两个文件体积较大 / 有其自身许可，请勿提交到本仓库（已被 `.gitignore` 排除）。

## 使用方法

### 1. 提取物种级特征库（需先准备数据）

```bash
python -m codes.ecological_analysis
```

对全部 200 类（每类若干张）提取特征并聚合，输出 `output/species_features_traits.csv` 及相关性图表。

### 2. 启动图形界面

```bash
python -m streamlit run app.py
```

浏览器打开 http://localhost:8501 ，包含「数据集浏览与统计」「图像分析与识别」「生态机理解释」三个页面。

### 3. 小样本跑通流水线（可选）

```bash
python -m codes.main
```

## 复现性

- 全局随机种子在 `codes/config.py` 中统一配置（`SEED = 42`），抽样、聚类初始化均固定。
- 若界面报 `.streamlit` 权限错误，可加参数：`python -m streamlit run app.py --browser.gatherUsageStats false`。

## 说明

- 本项目定位是**图像视觉分析**而非大模型分类：特征提取与分割全部基于 OpenCV 基础算子。
- 相关性分析结论为「相关而非因果」，仅供生态机理探讨参考。
