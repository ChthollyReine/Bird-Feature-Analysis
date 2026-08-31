"""第 7 步：生态指标映射与机理解释

把图像图案特征按物种聚合，与 AVONET 形态/生态性状做相关性分析与分组比较
输出可解释结论与图表

用法：
    python -m codes.ecological_analysis
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from . import matplotlib_font 
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

from . import config
from . import data_loader
from . import preprocessing
from . import segmentation
from . import ecological
from .features import extract_all


# 连续性/有序性状
CONTINUOUS_TRAITS = [
    "Mass", "Beak.Length_Culmen", "Beak.Width", "Beak.Depth",
    "Tarsus.Length", "Wing.Length", "Hand-Wing.Index", "Tail.Length",
    "Migration", "Range.Size", "Centroid.Latitude",
]

# 类别性状（用于分组比较）
CATEGORICAL_TRAITS = ["Habitat", "Trophic.Niche", "Primary.Lifestyle"]

# 生态指标估计：连续型形态/生态指标（用于上传图的生态指标估计）
ESTIMATE_NUMERIC_TRAITS = [
    "Mass", "Wing.Length", "Hand-Wing.Index",
    "Beak.Length_Culmen", "Beak.Width", "Beak.Depth",
    "Tarsus.Length", "Tail.Length",
]

# 生态指标估计：类别型生态指标
ESTIMATE_CATEGORICAL_TRAITS = ["Habitat", "Trophic.Niche", "Primary.Lifestyle", "Migration"]

# 用于展示/解读的关键图案特征
KEY_FEATURES = [
    "sym_lr", "sym_tb", "sym_axis_corr", "sym_bilateral_texture_corr",
    "fractal_dim", "foreground_ratio",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_entropy", "glcm_correlation",
    "color_h_mean", "color_s_mean", "color_s_std", "color_v_mean",
    "shape_circularity", "shape_aspect_ratio",
]


def extract_species_features(per_class=5, seed=config.SEED, verbose=True):
    """对全部 200 类各抽 per_class 张，聚合为每类特征均值。

    返回 species_df(index=class_id, columns=特征均值)
    """
    ds = data_loader.CUBDataset()
    subset = ds.sample_all_classes(per_class=per_class, seed=seed)

    accum = defaultdict(list)
    for i, (img_id, label) in enumerate(subset):
        img = ds.load_image(img_id)
        bbox = ds.bounding_box(img_id)
        img, bbox = preprocessing.preprocess(img, bbox=bbox)
        mask, _ = segmentation.segment(img, bbox=bbox)
        feats = extract_all(img, mask=mask)
        accum[label].append(feats)
        if verbose and (i + 1) % 200 == 0:
            print(f"   processed {i + 1}/{len(subset)}")

    rows = {label: pd.DataFrame(feats).mean() for label, feats in sorted(accum.items())}
    species_df = pd.DataFrame(rows).T
    species_df.index.name = "class_id"
    return species_df


def build_joined(species_df):
    """把物种级特征与 AVONET 性状合并为一张表"""
    avonet = ecological.load_avonet()
    mapping = ecological.build_mapping_frame()[["class_id", "cub_name", "scientific_name"]]
    joined = (
        species_df.reset_index()
        .merge(mapping, on="class_id", how="left")
        .merge(avonet, left_on="scientific_name", right_on="Species1", how="left")
    )
    return joined


def correlate(joined, feature_cols):
    """Spearman 相关性：连续性状 X 图案特征，返回按 p 值排序的长表"""
    results = []
    for trait in CONTINUOUS_TRAITS:
        if trait not in joined.columns:
            continue
        tv = pd.to_numeric(joined[trait], errors="coerce")
        for f in feature_cols:
            fv = pd.to_numeric(joined[f], errors="coerce")
            mask = tv.notna() & fv.notna()
            if mask.sum() < 10:
                continue
            if np.std(tv[mask]) < 1e-12 or np.std(fv[mask]) < 1e-12:
                continue  # 常量序列无相关定义
            rho, p = stats.spearmanr(tv[mask], fv[mask])
            results.append({"trait": trait, "feature": f, "rho": float(rho),
                            "p": float(p), "n": int(mask.sum())})
    return pd.DataFrame(results).sort_values("p")


def group_compare(joined):
    """分组比较:关键特征在各类别性状下的差异(Kruskal-Wallis 检验)"""     
    rows = []
    for trait in CATEGORICAL_TRAITS:
        if trait not in joined.columns:
            continue
        for f in KEY_FEATURES:
            groups = []
            for _, sub in joined.groupby(trait, dropna=True):
                fv = pd.to_numeric(sub[f], errors="coerce").dropna()
                if len(fv) >= 3:
                    groups.append(fv.values)
            if len(groups) < 2:
                continue
            stat, p = stats.kruskal(*groups)
            rows.append({"trait": trait, "feature": f, "kruskal_p": float(p),
                         "n_groups": len(groups)})
    return pd.DataFrame(rows).sort_values("kruskal_p")


def _save_heatmap(joined, feature_cols, path):
    """保存 关键特征 X 连续性状 的 Spearman 相关热图"""
    feats = [f for f in KEY_FEATURES if f in feature_cols]
    traits = [t for t in CONTINUOUS_TRAITS if t in joined.columns]
    mat = np.full((len(feats), len(traits)), np.nan)
    for i, f in enumerate(feats):
        for j, t in enumerate(traits):
            tv = pd.to_numeric(joined[t], errors="coerce")
            fv = pd.to_numeric(joined[f], errors="coerce")
            mask = tv.notna() & fv.notna()
            if mask.sum() >= 10:
                mat[i, j] = stats.spearmanr(tv[mask], fv[mask])[0]

    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels(traits, rotation=45, ha="right")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    plt.colorbar(im, ax=ax, label="Spearman rho")
    ax.set_title("Pattern features vs ecological traits (Spearman)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_scatter(joined, path):
    """示例散点：分形复杂度 vs 体重、左右对称 vs 迁徙"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    mass = pd.to_numeric(joined["Mass"], errors="coerce")
    fd = pd.to_numeric(joined["fractal_dim"], errors="coerce")
    axes[0].scatter(mass, fd, alpha=0.6)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Mass (g, log)")
    axes[0].set_ylabel("Fractal dimension")

    mig = pd.to_numeric(joined["Migration"], errors="coerce")
    sym = pd.to_numeric(joined["sym_lr"], errors="coerce")
    axes[1].scatter(mig, sym, alpha=0.6)
    axes[1].set_xlabel("Migration")
    axes[1].set_ylabel("Left-right symmetry")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(per_class=5):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== 1. 提取物种级图案特征（全部 200 类）==")
    species_df = extract_species_features(per_class=per_class)
    feature_cols = list(species_df.columns)
    print(f"   species features shape: {species_df.shape}")

    print("== 2. 关联 AVONET 性状 ==")
    joined = build_joined(species_df)
    out_csv = config.OUTPUT_DIR / "species_features_traits.csv"
    joined.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"   已保存: {out_csv}")

    print("== 3. 相关性分析(Spearman,特征 X 性状)==")
    corr = correlate(joined, feature_cols)
    sig = corr[corr["p"] < 0.05]
    print(f"   显著相关对(p<0.05): {len(sig)} / {len(corr)}")
    print(sig[["trait", "feature", "rho", "p", "n"]].head(20).to_string(index=False))

    print("== 4. 分组比较(Kruskal-Wallis)==")
    gc = group_compare(joined)
    sig_gc = gc[gc["kruskal_p"] < 0.05]
    print(sig_gc[["trait", "feature", "kruskal_p", "n_groups"]].head(15).to_string(index=False))

    print("== 5. 绘图 ==")
    _save_heatmap(joined, feature_cols, config.OUTPUT_DIR / "eco_corr_heatmap.png")
    _save_scatter(joined, config.OUTPUT_DIR / "eco_scatter.png")
    print("完成。输出目录:", config.OUTPUT_DIR)
    return joined, corr, gc


def _default_feature_names():
    """从空样例图推导全部特征名"""
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    return list(extract_all(img, None).keys())


FEATURE_NAMES = _default_feature_names()


def load_feature_library(csv_path=None):
    """加载物种特征库，返回 (df, feature_cols, scaler)

    df 即 species_features_traits.csv(200 行：特征 + AVONET 性状)
    feature_cols 为 FEATURE_NAMES 中存在的特征列名
    scaler 为 StandardScaler,已对特征列进行 fit
    """
    csv_path = str(csv_path or (config.OUTPUT_DIR / "species_features_traits.csv"))
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    X = df[feature_cols].astype(float).values
    scaler = StandardScaler().fit(X)
    return df, feature_cols, scaler


def estimate_ecological_traits(feature_dict, df=None, feature_cols=None, scaler=None, top_k=20):
    """基于特征相似样本的生态指标估计。

    在标准化特征空间里找与上传图最接近的 top_k 个物种，对其生态指标做统计：
    - 连续指标 -> 均值 + 范围（min~max）
    - 类别指标 -> 众数 + 占比

    返回 (neighbors, numeric, categorical)：
      neighbors:    最接近的 top_k 个物种 DataFrame（含 distance 列）
      numeric:      {trait: {"mean", "min", "max"}}
      categorical:  {trait: {"mode", "ratio"}}
    """
    from scipy.spatial.distance import cdist

    if df is None:
        df, feature_cols, scaler = load_feature_library()

    q = np.array([[float(feature_dict[k]) for k in feature_cols]])
    qs = scaler.transform(q)
    Xs = scaler.transform(df[feature_cols].astype(float).values)
    d = cdist(qs, Xs, metric="euclidean").ravel()

    order = np.argsort(d)[:top_k]
    neighbors = df.iloc[order].copy()
    neighbors["distance"] = d[order]

    numeric = {}
    for trait in ESTIMATE_NUMERIC_TRAITS:
        if trait not in neighbors.columns:
            continue
        vals = pd.to_numeric(neighbors[trait], errors="coerce").dropna()
        if vals.empty:
            continue
        numeric[trait] = {
            "mean": float(vals.mean()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    categorical = {}
    for trait in ESTIMATE_CATEGORICAL_TRAITS:
        if trait not in neighbors.columns:
            continue
        s = neighbors[trait].dropna()
        if s.empty:
            continue
        counts = s.value_counts()
        categorical[trait] = {
            "mode": str(counts.index[0]),
            "ratio": float(counts.iloc[0] / len(s)),
        }

    return neighbors, numeric, categorical


if __name__ == "__main__":
    run()
