"""对称性特征:对称轴检测与两边纹理相关性

量化动物体表图案的对称程度，用于生态适应性的形态分析
"""
import numpy as np
import cv2

from . import texture


def _half_corr(gray, fg=None, axis="vertical"):
    """计算左右(vertical 轴)或上下(horizontal 轴)两半的归一化互相关系数。

    fg: 可选布尔前景掩膜，传入时仅统计翻转后两侧均为前景的像素，排除背景干扰。
    """
    h, w = gray.shape
    if axis == "vertical":
        half = w // 2
        left = gray[:, :half].astype(np.float64)
        right = np.fliplr(gray[:, half:half * 2].astype(np.float64))
        if fg is not None:
            lm = fg[:, :half]
            rm = np.fliplr(fg[:, half:half * 2])
            sel = lm & rm
            left = left[sel]
            right = right[sel]
        else:
            left = left.ravel()
            right = right.ravel()
    else:
        half = h // 2
        left = gray[:half, :].astype(np.float64)
        right = np.flipud(gray[half:half * 2, :].astype(np.float64))
        if fg is not None:
            lm = fg[:half, :]
            rm = np.flipud(fg[half:half * 2, :])
            sel = lm & rm
            left = left[sel]
            right = right[sel]
        else:
            left = left.ravel()
            right = right.ravel()

    if left.size < 2 or right.size < 2:
        return 0.0

    left = left - left.mean()
    right = right - right.mean()
    denom = (left.std() * right.std())
    if denom < 1e-12:
        return 0.0
    return float((left * right).mean() / denom)


def _bilateral_texture_corr(gray, fg=None):
    """两边纹理相关性：左右两半各自提取 GLCM 特征向量后计算 Pearson 相关""" 
    h, w = gray.shape
    half = w // 2
    left = gray[:, :half]
    right = gray[:, half:half * 2]
    lm = rm = None
    if fg is not None:
        lm = fg[:, :half]
        rm = fg[:, half:half * 2]

    fl = texture.glcm_features(left, mask=lm)
    fr = texture.glcm_features(right, mask=rm)
    keys = sorted(fl.keys())
    vl = np.array([fl[k] for k in keys])
    vr = np.array([fr[k] for k in keys])

    if np.std(vl) < 1e-12 or np.std(vr) < 1e-12:
        return 0.0
    return float(np.corrcoef(vl, vr)[0, 1])


def symmetry_axis(gray, fg=None):
    """粗略检测对称轴：以左右翻转互相关峰值位置估计竖直对称轴偏移"""
    if fg is not None and fg.any():
        # 裁剪到前景外接框，减少背景对对称轴检测的影响
        ys, xs = np.where(fg)
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        gray = gray[y0:y1, x0:x1]
    h, w = gray.shape
    if w < 2 or h < 2:
        return 0, 0.0
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


def symmetry_features(gray, mask=None):
    """返回对称性特征字典。

    mask: 可选前景掩膜（255=前景），传入时仅对鸟主体区域计算对称性。
    """
    fg = None
    if mask is not None and mask.any():
        fg = mask > 0
    shift, axis_corr = symmetry_axis(gray, fg)
    lr = _half_corr(gray, fg, axis="vertical")
    tb = _half_corr(gray, fg, axis="horizontal")
    bilat = _bilateral_texture_corr(gray, fg)
    return {
        "sym_lr": lr,
        "sym_tb": tb,
        "sym_axis_shift": float(shift),
        "sym_axis_corr": float(axis_corr),
        "sym_bilateral_texture_corr": bilat,
    }
