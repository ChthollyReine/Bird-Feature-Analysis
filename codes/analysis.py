"""降维与聚类:PCA 主成分分析 + KMeans 空间聚类图谱。"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from . import config


def build_matrix(feature_dicts, feature_names=None):
    """把特征字典列表组装成 ndarray 特征矩阵。

    feature_names: 固定键顺序;None 时取第一个样本的键（要求所有样本键一致）。
    """
    if feature_names is None:
        feature_names = list(feature_dicts[0].keys())
    return np.array([[d[k] for k in feature_names] for d in feature_dicts]), feature_names


def pca_reduce(matrix, n_components=None):
    """标准化 + PCA 降维，返回 (降维坐标, PCA 对象)。"""
    n_components = n_components or config.ANALYSIS["n_components"]
    X = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=n_components, random_state=config.SEED)
    return pca.fit_transform(X), pca


def cluster(matrix, n_clusters=None):
    """KMeans 聚类，返回 (聚类标签, KMeans 对象)。"""
    n_clusters = n_clusters or config.ANALYSIS["n_clusters"]
    X = StandardScaler().fit_transform(matrix)
    km = KMeans(n_clusters=n_clusters, random_state=config.SEED, n_init=10)
    labels = km.fit_predict(X)
    return labels, km
