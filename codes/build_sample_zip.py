"""从 archive.zip 为每个物种抽取 1 张图片，生成精简 sample_images.zip。

用途：完整数据集（archive.zip，约 1.49GB）不提交到 Git，云端/演示环境用这份精简包
（200 张图 + 对应标注）即可在界面展示缩略图。

用法：
    python -m codes.build_sample_zip
"""
import zipfile
from collections import OrderedDict

from . import config


def build(src=None, dst=None):
    src = src or config.DATA_ZIP
    dst = dst or config.SAMPLE_ZIP

    if not src.exists():
        raise FileNotFoundError(f"未找到 {src}，请先下载 CUB-200-2011 数据集")

    z = zipfile.ZipFile(src)
    prefix = config.CUB_PREFIX

    images_lines = z.read(f"{prefix}/images.txt").decode("utf-8").splitlines()
    bbox_lines = z.read(f"{prefix}/bounding_boxes.txt").decode("utf-8").splitlines()
    label_lines = z.read(f"{prefix}/image_class_labels.txt").decode("utf-8").splitlines()
    classes_lines = z.read(f"{prefix}/classes.txt").decode("utf-8").splitlines()

    # image_id -> [rel_path, bbox_line, label_line]
    info = {}
    for line in images_lines:
        iid, rel = line.split(" ", 1)
        info[int(iid)] = [rel, None, None]
    for line in bbox_lines:
        info[int(line.split()[0])][1] = line
    for line in label_lines:
        info[int(line.split()[0])][2] = line

    # 每个类别取 image_id 最小的一张
    first_per_class = OrderedDict()
    for iid in sorted(info):
        label = int(info[iid][2].split()[1])
        first_per_class.setdefault(label, iid)

    picked_ids = set(first_per_class.values())

    out = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
    out.writestr(f"{prefix}/classes.txt", "\n".join(classes_lines) + "\n")
    out.writestr(f"{prefix}/images.txt",
                 "\n".join(l for l in images_lines if int(l.split()[0]) in picked_ids) + "\n")
    out.writestr(f"{prefix}/bounding_boxes.txt",
                 "\n".join(l for l in bbox_lines if int(l.split()[0]) in picked_ids) + "\n")
    out.writestr(f"{prefix}/image_class_labels.txt",
                 "\n".join(l for l in label_lines if int(l.split()[0]) in picked_ids) + "\n")

    for iid in sorted(picked_ids):
        rel = info[iid][0]
        out.writestr(f"{prefix}/images/{rel}", z.read(f"{prefix}/images/{rel}"))

    out.close()
    z.close()
    return len(picked_ids)


if __name__ == "__main__":
    n = build()
    print(f"完成：已抽取 {n} 张图片 -> {config.SAMPLE_ZIP}")
