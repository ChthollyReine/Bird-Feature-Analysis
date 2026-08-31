"""CUB-200-2011 数据加载:直接从 archive.zip 读取，不落盘解压
"""
import zipfile

import numpy as np
import cv2

from . import config


class CUBDataset:
    """从 archive.zip 读取 CUB-200-2011 图像与标注"""

    def __init__(self, zip_path=None):
        self.zip_path = str(zip_path or config.DATA_ZIP)
        self._zip = zipfile.ZipFile(self.zip_path)

        self.id2path = {}
        for line in self._read_text("images.txt"):
            img_id, rel = line.split(" ", 1)
            self.id2path[int(img_id)] = rel

        # image_id -> (x, y, w, h) 边界框
        self.id2bbox = {}
        for line in self._read_text("bounding_boxes.txt"):
            parts = line.split()
            img_id = int(parts[0])
            self.id2bbox[img_id] = tuple(int(float(v)) for v in parts[1:5])

        # image_id -> 类别号（1..200）
        self.id2label = {}
        for line in self._read_text("image_class_labels.txt"):
            img_id, label = line.split()
            self.id2label[int(img_id)] = int(label)

    def _read_text(self, name):
        path = f"{config.CUB_PREFIX}/{name}"
        return self._zip.read(path).decode("utf-8").splitlines()

    @property
    def image_ids(self):
        return list(self.id2path.keys())

    def load_image(self, img_id):
        """按 image_id 从 zip 中读取并解码为 BGR ndarray"""
        rel = self.id2path[img_id]
        path = f"{config.CUB_PREFIX}/images/{rel}"
        data = self._zip.read(path)
        buf = np.frombuffer(data, np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def bounding_box(self, img_id):
        """返回 (x, y, w, h)"""
        return self.id2bbox.get(img_id)

    def label(self, img_id):
        return self.id2label.get(img_id)

    def sample(self, n_classes=config.N_CLASSES, per_class=config.PER_CLASS, seed=config.SEED):
        """抽样一个小规模子集：前 n_classes 个类别，每类 per_class 张

        返回 [(img_id, label), ...]
        """
        rng = np.random.default_rng(seed)
        by_label = {}
        for img_id, label in self.id2label.items():
            by_label.setdefault(label, []).append(img_id)

        picked = []
        for label in sorted(by_label)[:n_classes]:
            ids = by_label[label]
            chosen = rng.choice(sorted(ids), size=min(per_class, len(ids)), replace=False)
            picked.extend((int(i), label) for i in chosen)
        return picked

    def sample_all_classes(self, per_class=5, seed=config.SEED):
        """对全部 200 类各抽 per_class 张，返回 [(img_id, label), ...]

        用于物种级特征聚合与生态分析
        """
        rng = np.random.default_rng(seed)
        by_label = {}
        for img_id, label in self.id2label.items():
            by_label.setdefault(label, []).append(img_id)

        picked = []
        for label in sorted(by_label):
            ids = sorted(by_label[label])
            chosen = rng.choice(ids, size=min(per_class, len(ids)), replace=False)
            picked.extend((int(i), label) for i in chosen)
        return picked
