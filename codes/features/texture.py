"""纹理特征：灰度共生矩阵（GLCM）+ Haralick 统计量。

纯 numpy 实现，不用 skimage，符合"OpenCV 基础算子组合"约束。
"""
import numpy as np

from .. import config


def _quantize(gray, levels):
    """灰度量化到 [0, levels-1]。"""
    gray = gray.astype(np.float32)
    lo, hi = gray.min(), gray.max()
    if hi - lo < 1e-6:
        return np.zeros_like(gray, dtype=np.int64)
    q = np.floor((gray - lo) / (hi - lo) * (levels - 1)).astype(np.int64)
    return np.clip(q, 0, levels - 1)


def _cooccurrence(q, levels, dx, dy):
    """计算指定偏移 (dx, dy) 下的共生矩阵。"""
    h, w = q.shape
    # 源与目标区域的对齐切片
    src = q[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    dst = q[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
    idx = src.ravel() * levels + dst.ravel()
    mat = np.bincount(idx, minlength=levels * levels).reshape(levels, levels)
    return mat


def _haralick(mat):
    """从共生矩阵计算 Haralick 纹理统计量。"""
    total = mat.sum()
    if total == 0:
        return dict(contrast=0.0, dissimilarity=0.0, homogeneity=0.0,
                    energy=0.0, correlation=0.0, entropy=0.0)
    p = mat.astype(np.float64) / total

    i = np.arange(p.shape[0], dtype=np.float64)
    j = np.arange(p.shape[1], dtype=np.float64)
    ii, jj = np.meshgrid(i, j, indexing="ij")

    contrast = float(((ii - jj) ** 2 * p).sum())
    dissimilarity = float((np.abs(ii - jj) * p).sum())
    homogeneity = float((p / (1.0 + (ii - jj) ** 2)).sum())
    energy = float((p ** 2).sum())

    mu_i = float((ii * p).sum())
    mu_j = float((jj * p).sum())
    si = float(((ii - mu_i) ** 2 * p).sum()) ** 0.5
    sj = float(((jj - mu_j) ** 2 * p).sum()) ** 0.5
    if si < 1e-12 or sj < 1e-12:
        correlation = 0.0
    else:
        correlation = float((((ii - mu_i) * (jj - mu_j) * p).sum()) / (si * sj))

    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = float(-(p * np.log(p + 1e-12)).sum())

    return dict(contrast=contrast, dissimilarity=dissimilarity, homogeneity=homogeneity,
                energy=energy, correlation=correlation, entropy=entropy)


def glcm_features(gray):
    """返回多角度/多距离平均后的 GLCM 特征。"""
    cfg = config.GLCM
    q = _quantize(gray, cfg["levels"])
    acc = {k: 0.0 for k in
           ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "entropy")}
    n = 0
    for d in cfg["distances"]:
        for ang in cfg["angles"]:
            rad = np.deg2rad(ang)
            dx = int(round(d * np.cos(rad)))
            dy = int(round(d * np.sin(rad)))
            mat = _cooccurrence(q, cfg["levels"], dx, dy)
            feats = _haralick(mat)
            for k in acc:
                acc[k] += feats[k]
            n += 1
    return {f"glcm_{k}": v / n for k, v in acc.items()}
