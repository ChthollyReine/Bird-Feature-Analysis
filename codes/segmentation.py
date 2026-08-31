"""传统 / 自适应分割：前景（鸟主体）掩膜提取。

只用 OpenCV 基础算子(Otsu、形态学、轮廓)，不使用 SAM/YOLO。
"""
import cv2
import numpy as np

from . import config


def foreground_mask(img_bgr):
    """Otsu 阈值 + 形态学开运算，得到前景二值掩膜。"""
    p = config.SEGMENT
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (p["blur_ksize"], p["blur_ksize"]), 0)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (p["morph_open_ksize"], p["morph_open_ksize"])
    )
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 保留最大连通域作为鸟主体，剔除背景噪点
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(opened)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    return mask


def apply_mask(img_bgr, mask):
    """用掩膜抠出前景，背景置 0。"""
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask)


def segment(img_bgr):
    """分割主入口：返回 (掩膜, 抠图结果)。"""
    mask = foreground_mask(img_bgr)
    masked = apply_mask(img_bgr, mask)
    return mask, masked
