# services/clustering_service.py

"""
Clustering Service
Quản lý việc chạy DBSCAN và xử lý kết quả:
- Chạy DBSCAN
- Lấy danh sách cluster
- Phát hiện outliers
- Tạo thống kê cluster
"""

import pandas as pd

from config import (
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    OUTLIER_LABEL,
    OUTLIER_PREVIEW_ROWS
)

from models.dbscan_model import dbscan_pipeline

from services.data_service import get_dbscan_data


# ===============================
# RUN DBSCAN
# ===============================

def run_dbscan():
    """
    Chạy thuật toán DBSCAN
    """

    data, scaled_features = get_dbscan_data()

    clustered_data, outliers, summary = dbscan_pipeline(
        data,
        scaled_features
    )

    return clustered_data, outliers, summary


# ===============================
# CLUSTER DISTRIBUTION
# ===============================

def get_cluster_distribution(clustered_data):
    """
    Thống kê số giao dịch trong mỗi cluster
    """

    cluster_counts = clustered_data["Cluster"].value_counts()

    cluster_dict = cluster_counts.to_dict()

    return cluster_dict


# ===============================
# GET OUTLIERS
# ===============================

def get_outliers(clustered_data):
    """
    Lấy các giao dịch bị đánh dấu là outlier
    """

    outliers = clustered_data[
        clustered_data["Cluster"] == OUTLIER_LABEL
    ]

    return outliers


# ===============================
# OUTLIER PREVIEW
# ===============================

def get_outlier_preview(clustered_data):
    """
    Lấy một phần outlier để hiển thị UI
    """

    outliers = get_outliers(clustered_data)

    preview = outliers.head(OUTLIER_PREVIEW_ROWS)

    return preview


# ===============================
# CLUSTER SUMMARY
# ===============================

def get_cluster_summary(clustered_data):
    """
    Tạo bảng thống kê cluster
    """

    cluster_counts = clustered_data["Cluster"].value_counts()

    summary = pd.DataFrame({
        "Cluster": cluster_counts.index,
        "Transactions": cluster_counts.values
    })

    summary = summary.sort_values(by="Cluster")

    return summary


# ===============================
# FULL CLUSTERING PIPELINE
# ===============================

def clustering_analysis():
    """
    Pipeline hoàn chỉnh cho clustering
    """

    clustered_data, outliers, summary = run_dbscan()

    cluster_distribution = get_cluster_distribution(clustered_data)

    outlier_preview = get_outlier_preview(clustered_data)

    cluster_summary = get_cluster_summary(clustered_data)

    result = {
        "clustered_data": clustered_data,
        "outliers": outliers,
        "cluster_distribution": cluster_distribution,
        "outlier_preview": outlier_preview,
        "cluster_summary": cluster_summary
    }

    return result