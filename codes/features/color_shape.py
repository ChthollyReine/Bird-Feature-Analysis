"""颜色与形貌特征:HSV 直方图统计 + 轮廓几何描述"""
import cv2
import numpy as np

from .. import config


def color_features(img_bgr):
    """HSV 各通道归一化直方图 + 均值/标准差"""
    cfg = config.COLOR
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = {}

    bins = (cfg["h_bins"], cfg["s_bins"], cfg["v_bins"])
    names = ("h", "s", "v")
    for ch, name, n_bins in zip(cv2.split(hsv), names, bins):
        hist = cv2.calcHist([ch], [0], None, [n_bins], [0, 256])
        hist = hist.ravel()
        if hist.sum() > 0:
            hist = hist / hist.sum()
        for i, val in enumerate(hist):
            feats[f"color_{name}_bin{i}"] = float(val)
        feats[f"color_{name}_mean"] = float(ch.mean())
        feats[f"color_{name}_std"] = float(ch.std())

    return feats


def shape_features(mask):
    """基于前景掩膜轮廓的形貌特征

    mask 为 None 时返回空特征 , 保证特征键一致性 
    """
    keys = ["shape_area", "shape_perimeter", "shape_circularity",
            "shape_aspect_ratio", "shape_solidity", "shape_extent"]
    if mask is None or not mask.any():
        return {k: 0.0 for k in keys}

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {k: 0.0 for k in keys}

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    if perimeter <= 0:
        return {k: 0.0 for k in keys}

    circularity = 4.0 * np.pi * area / (perimeter ** 2)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(h) / float(w) if w > 0 else 0.0
    extent = area / float(w * h) if w * h > 0 else 0.0

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    return {
        "shape_area": area,
        "shape_perimeter": perimeter,
        "shape_circularity": circularity,
        "shape_aspect_ratio": aspect_ratio,
        "shape_solidity": solidity,
        "shape_extent": extent,
    }
