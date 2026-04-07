# models/kmeans_model.py

import pandas as pd
from sklearn.cluster import KMeans


def run_kmeans(scaled_features, n_clusters=3, random_state=42):
    """
    Chạy thuật toán K-Means để phân cụm dữ liệu

    Parameters:
    scaled_features : dữ liệu đã chuẩn hóa
    n_clusters : số cụm cần tạo (mặc định = 3)
    random_state : để có kết quả nhất quán

    Returns:
    clusters : nhãn cụm của từng giao dịch
    model : model K-Means đã được huấn luyện
    """

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = model.fit_predict(scaled_features)

    return clusters, model


def add_cluster_labels(data, clusters):
    """
    Thêm cột KMeans_Cluster vào dataset gốc
    """

    data_with_cluster = data.copy()
    data_with_cluster["KMeans_Cluster"] = clusters

    return data_with_cluster


def cluster_summary(data_with_cluster):
    """
    Thống kê số lượng giao dịch trong mỗi cụm
    """

    summary = data_with_cluster["KMeans_Cluster"].value_counts().sort_index()

    return summary


def get_cluster_centers(model, scaled_features):
    """
    Lấy tâm điểm của mỗi cụm
    """

    centers = model.cluster_centers_

    return centers


def kmeans_pipeline(data, scaled_features, n_clusters=3):
    """
    Pipeline hoàn chỉnh cho K-Means
    """

    clusters, model = run_kmeans(scaled_features, n_clusters=n_clusters)

    data_clustered = add_cluster_labels(data, clusters)

    summary = cluster_summary(data_clustered)

    centers = get_cluster_centers(model, scaled_features)

    # Tính inertia (tổng khoảng cách từ điểm đến tâm cụm)
    inertia = model.inertia_

    return data_clustered, summary, centers, inertia
