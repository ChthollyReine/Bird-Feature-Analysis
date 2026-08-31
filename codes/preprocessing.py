"""针对性 DIP 去噪 / 滤波 / 几何增强 / 对齐。
核心只用 OpenCV 基础算子实现
"""
import cv2
import numpy as np

from . import config


def denoise(img_bgr):
    """双边滤波去噪：保留羽毛纹理边缘的同时平滑噪声"""
    p = config.PREPROCESS
    return cv2.bilateralFilter(
        img_bgr, p["bilateral_d"], p["bilateral_sigma_color"], p["bilateral_sigma_space"]
    )


def enhance(img_bgr):
    """对比度增强(CLAHE 作用于亮度通道)"""
    p = config.PREPROCESS
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=p["clahe_clip"], tileGridSize=p["clahe_grid"])
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def align(img_bgr, bbox=None):
    """几何对齐：裁剪主体 + 仿射归一化到统一尺寸。

    返回 (resized, bbox_out)：bbox_out 为原 bbox 映射到 256×256 归一化图后的坐标，
    用于后续分割的 GrabCut 初始化；无 bbox 时 bbox_out 为 None（分割走矩形初始化）。
    """
    p = config.PREPROCESS
    # 骨架阶段：用边界框裁剪鸟主体；后续可扩展为基于关键点的 Affine/Perspective
    # 对齐（摆正朝向），以支撑对称轴检测。
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        h_img, w_img = img_bgr.shape[:2]
        # 外扩边界框，给分割保留背景上下文，避免 GrabCut 无背景可参照
        pad = p["bbox_pad"]
        pad_x = int(w * pad)
        pad_y = int(h * pad)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w_img, x + w + pad_x)
        y1 = min(h_img, y + h + pad_y)
        img_bgr = img_bgr[y0:y1, x0:x1]
        # 原 bbox 映射到裁剪后的坐标
        nx, ny, nw, nh = x - x0, y - y0, w, h

        # 归一化到固定尺寸，便于后续统一计算
        resized = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
        sx = 256.0 / (x1 - x0)
        sy = 256.0 / (y1 - y0)
        bbox_out = (nx * sx, ny * sy, nw * sx, nh * sy)
    else:
        resized = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
        bbox_out = None

    return resized, bbox_out


def preprocess(img_bgr, bbox=None):
    """预处理主入口:去噪 -> 增强 -> 对齐。

    返回 (img, bbox_out)：bbox_out 为映射到归一化图后的主体边界框，供分割使用。
    """
    img = denoise(img_bgr)
    img = enhance(img)
    img, bbox_out = align(img, bbox=bbox)
    return img, bbox_out
