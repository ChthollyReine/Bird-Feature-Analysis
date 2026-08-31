"""分形维数特征：Box-counting 方法估计图案复杂度。"""
import numpy as np

from .. import config


def _count_boxes(binary, s):
    """向量化统计覆盖前景像素的 s×s 盒子数量。"""
    h, w = binary.shape
    h2 = ((h + s - 1) // s) * s
    w2 = ((w + s - 1) // s) * s
    if h2 == 0 or w2 == 0:
        return 0
    pad = np.zeros((h2, w2), dtype=bool)
    pad[:h, :w] = binary
    blocks = pad.reshape(h2 // s, s, w2 // s, s)
    return int(blocks.any(axis=(1, 3)).sum())


def box_counting_dimension(binary, box_sizes=None):
    """对二值图计算 Box-counting 分形维数 D。

    D = log N(s) / log(1/s)，对多个尺度做最小二乘拟合得到斜率。
    """
    if box_sizes is None:
        box_sizes = config.FRACTAL["box_sizes"]

    counts = [_count_boxes(binary, s) for s in box_sizes]

    x = np.log(1.0 / np.array(box_sizes, dtype=np.float64))
    y = np.log(np.array(counts, dtype=np.float64) + 1e-12)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def fractal_features(gray):
    """返回基于二值化图案的分形维数特征。"""
    thr = config.FRACTAL["threshold"]
    _, binary = _threshold(gray, thr)
    binary = binary.astype(bool)
    dim = box_counting_dimension(binary)
    # 前景占比作为补充复杂度/覆盖度指标
    coverage = float(binary.mean())
    return {"fractal_dim": dim, "foreground_ratio": coverage}


def _threshold(gray, thr):
    """简单阈值（无依赖实现，便于独立使用）。"""
    binary = (gray > thr).astype(np.uint8) * 255
    return binary, binary
