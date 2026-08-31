"""针对性 DIP 去噪 / 滤波 / 几何增强 / 对齐。
核心只用 OpenCV 基础算子实现(bilateralFilter、CLAHE、getPerspectiveTransform 等)，
不使用任何大模型。
"""
import cv2
import numpy as np

from . import config


def denoise(img_bgr):
    """双边滤波去噪：保留羽毛纹理边缘的同时平滑噪声。"""
    p = config.PREPROCESS
    return cv2.bilateralFilter(
        img_bgr, p["bilateral_d"], p["bilateral_sigma_color"], p["bilateral_sigma_space"]
    )


def enhance(img_bgr):
    """对比度增强(CLAHE 作用于亮度通道)。"""
    p = config.PREPROCESS
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=p["clahe_clip"], tileGridSize=p["clahe_grid"])
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def align(img_bgr, bbox=None):
    """几何对齐：裁剪主体 + 仿射归一化到统一尺寸。

    骨架阶段：用边界框裁剪鸟主体；后续可扩展为基于关键点的 Affine/Perspective
    对齐（摆正朝向），以支撑对称轴检测。
    """
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        h_img, w_img = img_bgr.shape[:2]
        x = max(0, x)
        y = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        img_bgr = img_bgr[y:y2, x:x2]

    # 归一化到固定尺寸，便于后续统一计算
    return cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)


def preprocess(img_bgr, bbox=None):
    """预处理主入口：去噪 -> 增强 -> 对齐。"""
    img = denoise(img_bgr)
    img = enhance(img)
    img = align(img, bbox=bbox)
    return img
