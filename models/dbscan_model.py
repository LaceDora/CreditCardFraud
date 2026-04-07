# models/dbscan_model.py

import pandas as pd
from sklearn.cluster import DBSCAN


def run_dbscan(scaled_features, eps=0.5, min_samples=5):
    """
    Chạy thuật toán DBSCAN để phân cụm dữ liệu

    Parameters:
    scaled_features : dữ liệu đã chuẩn hóa
    eps : bán kính tìm điểm lân cận
    min_samples : số điểm tối thiểu để tạo cụm

    Returns:
    clusters : nhãn cụm của từng giao dịch
    """

    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(scaled_features)

    return clusters


def add_cluster_labels(data, clusters):
    """
    Thêm cột cluster vào dataset gốc
    """

    data_with_cluster = data.copy()
    data_with_cluster["Cluster"] = clusters

    return data_with_cluster


def get_outliers(data_with_cluster):
    """
    Lấy các giao dịch bất thường (noise)
    DBSCAN đánh dấu noise bằng -1
    """

    outliers = data_with_cluster[data_with_cluster["Cluster"] == -1]

    return outliers


def cluster_summary(data_with_cluster):
    """
    Thống kê số lượng giao dịch trong mỗi cụm
    """

    summary = data_with_cluster["Cluster"].value_counts().sort_index()

    return summary


def dbscan_pipeline(data, scaled_features):
    """
    Pipeline hoàn chỉnh cho DBSCAN
    """

    clusters = run_dbscan(scaled_features)

    data_clustered = add_cluster_labels(data, clusters)

    outliers = get_outliers(data_clustered)

    summary = cluster_summary(data_clustered)

    return data_clustered, outliers, summary