"""传统 / 自适应分割：前景（鸟主体）掩膜提取。

主用 GrabCut（经典图割前景分割），Otsu 作为后备；只用 OpenCV 基础算子，不使用 SAM/YOLO。
"""
import cv2
import numpy as np

from . import config


def grabcut_mask(img_bgr, bbox=None, iter_count=None):
    """GrabCut 前景分割，返回二值前景掩膜（255=前景）。

    bbox: (x, y, w, h) 鸟主体边界框（与 img_bgr 同一坐标系）。提供时用 bbox 内部作为
          "可能前景"、外部作为"可能背景"初始化（先验强、更稳，避免误切鸟体）；
          未提供时退化为整图内缩 5% 的矩形初始化（用于无标注的上传图）。
    """
    iter_count = iter_count or config.SEGMENT.get("grabcut_iter", 5)
    h, w = img_bgr.shape[:2]

    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    if bbox is not None:
        x, y, bw, bh = [int(round(v)) for v in bbox]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        # bbox 内部 -> 可能前景；中心区域 -> 确定前景（鸟主体通常居中）
        mask[y1:y2, x1:x2] = cv2.GC_PR_FGD
        cx1 = x1 + (x2 - x1) // 4
        cy1 = y1 + (y2 - y1) // 4
        cx2 = x1 + 3 * (x2 - x1) // 4
        cy2 = y1 + 3 * (y2 - y1) // 4
        mask[cy1:cy2, cx1:cx2] = cv2.GC_FGD
        # 图像边界 -> 确定背景，为背景 GMM 提供可靠锚点
        b = 5
        mask[:b, :] = cv2.GC_BGD
        mask[-b:, :] = cv2.GC_BGD
        mask[:, :b] = cv2.GC_BGD
        mask[:, -b:] = cv2.GC_BGD
        cv2.grabCut(img_bgr, mask, None, bgd, fgd, iter_count, cv2.GC_INIT_WITH_MASK)
    else:
        m = max(2, int(0.05 * min(h, w)))
        rect = (m, m, w - 2 * m, h - 2 * m)
        cv2.grabCut(img_bgr, mask, rect, bgd, fgd, iter_count, cv2.GC_INIT_WITH_RECT)

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return fg


def _cleanup(mask):
    """形态学后处理：填洞、去噪，并保留最大连通域。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [largest], -1, 255, cv2.FILLED)
    return out


def _otsu_mask(img_bgr):
    """Otsu 阈值（后备方案）。"""
    p = config.SEGMENT
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (p["blur_ksize"], p["blur_ksize"]), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def foreground_mask(img_bgr, bbox=None):
    """前景掩膜：GrabCut + 后处理；若前景过小则退回 Otsu。"""
    mask = grabcut_mask(img_bgr, bbox=bbox)
    mask = _cleanup(mask)
    if cv2.countNonZero(mask) < 0.02 * mask.size:
        # GrabCut 结果异常（前景几乎为空），退回 Otsu
        mask = _cleanup(_otsu_mask(img_bgr))
    return mask


def apply_mask(img_bgr, mask):
    """用掩膜抠出前景，背景置 0。"""
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask)


def segment(img_bgr, bbox=None):
    """分割主入口：返回 (掩膜, 抠图结果)。

    bbox: 可选 (x, y, w, h)，与 img_bgr 同一坐标系，用于初始化 GrabCut。
    """
    mask = foreground_mask(img_bgr, bbox=bbox)
    masked = apply_mask(img_bgr, mask)
    return mask, masked
