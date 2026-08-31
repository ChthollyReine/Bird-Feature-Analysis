"""主流程：生态图像特征工程与轻量级量化评价。

流程：输入 -> 预处理 -> 分割 -> 特征提取 -> PCA/聚类 -> 生态映射
"""
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示环境保存图片
import matplotlib.pyplot as plt

from . import config
from . import data_loader
from . import preprocessing
from . import segmentation
from . import analysis
from .features import extract_all


def set_seed(seed=config.SEED):
    """固定随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)


def run():
    set_seed()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("== 1. 数据加载 ==")
    ds = data_loader.CUBDataset()
    subset = ds.sample()
    print(f"   抽样 {len(subset)} 张（{config.N_CLASSES} 类 × {config.PER_CLASS}）")

    print("== 2~4. 预处理 / 分割 / 特征提取 ==")
    feature_dicts = []
    labels = []
    for img_id, label in subset:
        img = ds.load_image(img_id)
        bbox = ds.bounding_box(img_id)
        img = preprocessing.preprocess(img, bbox=bbox)
        mask, _ = segmentation.segment(img)
        feats = extract_all(img, mask=mask)
        feature_dicts.append(feats)
        labels.append(label)
        print(f"   image {img_id} (class {label}): {len(feats)} 维特征")

    matrix, feature_names = analysis.build_matrix(feature_dicts)
    print(f"   特征矩阵 shape: {matrix.shape}")

    print("== 5. PCA 降维 ==")
    coords, pca = analysis.pca_reduce(matrix)
    print(f"   PCA 解释方差比: {np.round(pca.explained_variance_ratio_, 4)}")

    print("== 6. 聚类 ==")
    if config.ANALYSIS["n_clusters"]:
        cluster_labels, km = analysis.cluster(matrix)
        print(f"   KMeans 聚类数: {km.n_clusters}")
    else:
        cluster_labels = None

    print("== 7. 可视化与输出 ==")
    _save_scatter(coords, labels, config.OUTPUT_DIR / "pca_scatter.png")

    out_csv = config.OUTPUT_DIR / "features.csv"
    _save_features(feature_names, feature_dicts, labels, out_csv)

    print("完成。输出目录:", config.OUTPUT_DIR)
    return coords, labels, feature_names, feature_dicts


def _save_scatter(coords, labels, path):
    """保存 PCA 二维散点图，颜色按类别标注。"""
    plt.figure(figsize=(8, 6))
    for lb in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lb]
        plt.scatter(coords[idx, 0], coords[idx, 1], label=f"class {lb}", s=40, alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("CUB pattern features - PCA projection")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   散点图已保存: {path}")


def _save_features(feature_names, feature_dicts, labels, path):
    """把特征矩阵落盘为 CSV，便于后续复现与生态映射。"""
    header = ["image_id", "label"] + feature_names
    rows = []
    for i, (d, lb) in enumerate(zip(feature_dicts, labels)):
        rows.append([i, lb] + [d[k] for k in feature_names])
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"   特征 CSV 已保存: {path}")


if __name__ == "__main__":
    run()
