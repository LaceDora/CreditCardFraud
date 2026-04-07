import pandas as pd
import matplotlib.pyplot as plt

from .constants import (
    TARGET_COLUMN,
    AMOUNT_COLUMN,
    TIME_COLUMN,
    CLUSTER_COLUMN,
    OUTLIER_LABEL,
    RULE_TABLE_COLUMNS
)


# ==============================
# DATASET SUMMARY
# ==============================

def dataset_summary(data):
    """
    Trả về thông tin tổng quan của dataset
    """

    if data is None or data.empty:
        return {
            "total_transactions": 0,
            "fraud_transactions": 0,
            "normal_transactions": 0
        }

    total = len(data)
    fraud = int(data[TARGET_COLUMN].sum())
    normal = total - fraud

    summary = {
        "total_transactions": total,
        "fraud_transactions": fraud,
        "normal_transactions": normal
    }

    return summary


# ==============================
# TOP TRANSACTIONS
# ==============================

def get_top_transactions(data, n=10):
    """
    Lấy các giao dịch có Amount lớn nhất
    """

    if AMOUNT_COLUMN not in data.columns:
        return None

    top = data.sort_values(
        by=AMOUNT_COLUMN,
        ascending=False
    ).head(n)

    return top


# ==============================
# FRAUD RATIO
# ==============================

def fraud_ratio(data):
    """
    Tính tỷ lệ giao dịch gian lận
    """

    total = len(data)

    if total == 0:
        return 0

    fraud = data[TARGET_COLUMN].sum()

    ratio = fraud / total

    return round(ratio, 4)


# ==============================
# CLUSTER COUNTS
# ==============================

def cluster_counts(data):
    """
    Đếm số lượng điểm trong mỗi cluster
    """

    if CLUSTER_COLUMN not in data.columns:
        return {}

    counts = data[CLUSTER_COLUMN].value_counts().sort_index()

    return counts.to_dict()


# ==============================
# PLOT CLUSTERS
# ==============================

def plot_clusters(data):
    """
    Vẽ biểu đồ scatter cho DBSCAN
    """

    if CLUSTER_COLUMN not in data.columns:
        return None

    if AMOUNT_COLUMN not in data.columns or TIME_COLUMN not in data.columns:
        return None

    x = data[AMOUNT_COLUMN]
    y = data[TIME_COLUMN]
    labels = data[CLUSTER_COLUMN]

    plt.figure(figsize=(8, 5))

    scatter = plt.scatter(
        x,
        y,
        c=labels,
        cmap="viridis",
        alpha=0.6
    )

    plt.xlabel("Transaction Amount")
    plt.ylabel("Transaction Time")
    plt.title("DBSCAN Clustering Result")

    plt.colorbar(scatter)

    # Highlight outliers
    outliers = data[data[CLUSTER_COLUMN] == OUTLIER_LABEL]

    if not outliers.empty:
        plt.scatter(
            outliers[AMOUNT_COLUMN],
            outliers[TIME_COLUMN],
            color="red",
            label="Outliers"
        )
        plt.legend()

    return plt


# ==============================
# FORMAT RULES TABLE
# ==============================

def format_rules_table(rules):
    """
    Chuẩn bị bảng luật để hiển thị
    """

    for col in RULE_TABLE_COLUMNS:
        if col not in rules.columns:
            return None

    table = rules[RULE_TABLE_COLUMNS].copy()

    table = table.reset_index(drop=True)

    return table