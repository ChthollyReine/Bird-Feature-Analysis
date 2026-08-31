"""对称性特征：对称轴检测与两边纹理相关性。

量化动物体表图案的对称程度，用于生态适应性的形态分析。
"""
import numpy as np
import cv2

from . import texture


def _half_corr(gray, axis="vertical"):
    """计算左右（vertical 轴）或上下（horizontal 轴）两半的归一化互相关系数。"""
    h, w = gray.shape
    if axis == "vertical":
        half = w // 2
        left = gray[:, :half].astype(np.float64)
        right = np.fliplr(gray[:, half:half * 2].astype(np.float64))
    else:
        half = h // 2
        left = gray[:half, :].astype(np.float64)
        right = np.flipud(gray[half:half * 2, :].astype(np.float64))

    if left.size < 2 or right.size < 2:
        return 0.0

    left = left - left.mean()
    right = right - right.mean()
    denom = (left.std() * right.std())
    if denom < 1e-12:
        return 0.0
    return float((left * right).mean() / denom)


def _bilateral_texture_corr(gray):
    """两边纹理相关性：左右两半各自提取 GLCM 特征向量后计算 Pearson 相关。"""
    h, w = gray.shape
    half = w // 2
    left = gray[:, :half]
    right = gray[:, half:half * 2]

    fl = texture.glcm_features(left)
    fr = texture.glcm_features(right)
    keys = sorted(fl.keys())
    vl = np.array([fl[k] for k in keys])
    vr = np.array([fr[k] for k in keys])

    if np.std(vl) < 1e-12 or np.std(vr) < 1e-12:
        return 0.0
    return float(np.corrcoef(vl, vr)[0, 1])


def symmetry_axis(gray):
    """粗略检测对称轴：以左右翻转互相关峰值位置估计竖直对称轴偏移。"""
    h, w = gray.shape
    g = gray.astype(np.float64)
    best_corr, best_shift = -1.0, 0
    for shift in range(-w // 8, w // 8 + 1):
        m = w - abs(shift)
        a = g[:, max(0, shift):max(0, shift) + m]
        b = np.fliplr(g[:, max(0, -shift):max(0, -shift) + m])
        a = a - a.mean()
        b = b - b.mean()
        denom = a.std() * b.std()
        if denom < 1e-12:
            continue
        corr = (a * b).mean() / denom
        if corr > best_corr:
            best_corr, best_shift = corr, shift
    return best_shift, best_corr


def symmetry_features(gray):
    """返回对称性特征字典。"""
    shift, axis_corr = symmetry_axis(gray)
    lr = _half_corr(gray, axis="vertical")
    tb = _half_corr(gray, axis="horizontal")
    bilat = _bilateral_texture_corr(gray)
    return {
        "sym_lr": lr,
        "sym_tb": tb,
        "sym_axis_shift": float(shift),
        "sym_axis_corr": float(axis_corr),
        "sym_bilateral_texture_corr": bilat,
    }
