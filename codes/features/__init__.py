"""特征提取统一入口：组合纹理 / 分形 / 对称 / 颜色 / 形貌特征"""
from collections import OrderedDict

import cv2

from . import texture, fractal, symmetry, color_shape


def extract_all(img_bgr, mask=None):
    """提取一张图的所有特征，返回有序字典（键名固定，便于组装特征矩阵） 

    img_bgr: BGR 图（已预处理）
    mask:    可选前景掩膜
    """
    feats = OrderedDict()

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    feats.update(texture.glcm_features(gray, mask=mask))
    feats.update(fractal.fractal_features(gray, mask=mask))
    feats.update(symmetry.symmetry_features(gray, mask=mask))
    feats.update(color_shape.color_features(img_bgr, mask=mask))
    feats.update(color_shape.shape_features(mask))

    return feats
