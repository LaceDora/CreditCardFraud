# services/data_service.py

"""
Data Service
Quản lý toàn bộ thao tác với dataset:
- Load dữ liệu
- Thống kê fraud
- Preview dataset
- Lấy feature cho mô hình
"""

import pandas as pd

from config import (
    DATA_PATH,
    MAX_ROWS,
    TARGET_COLUMN,
    DATASET_PREVIEW_ROWS
)

from preprocessing.preprocess import preprocess_pipeline


# ===============================
# LOAD DATA
# ===============================

def load_dataset():
    """
    Đọc dataset từ file CSV
    """

    df = pd.read_csv(DATA_PATH, nrows=MAX_ROWS)

    return df


# ===============================
# PREPROCESS DATA
# ===============================

def get_processed_data():
    """
    Lấy dữ liệu đã preprocessing
    """

    X_scaled, y = preprocess_pipeline(DATA_PATH)

    return X_scaled, y


# ===============================
# DATASET PREVIEW
# ===============================

def get_dataset_preview():
    """
    Lấy preview dataset để hiển thị UI
    """

    df = load_dataset()

    preview = df.head(DATASET_PREVIEW_ROWS)

    return preview


# ===============================
# FRAUD SUMMARY
# ===============================

def get_fraud_summary():
    """
    Thống kê fraud dataset
    """

    df = load_dataset()

    total_transactions = len(df)

    fraud_transactions = int(df[TARGET_COLUMN].sum())

    normal_transactions = total_transactions - fraud_transactions

    fraud_rate = round((fraud_transactions / total_transactions) * 100, 4)

    summary = {
        "total": total_transactions,
        "fraud": fraud_transactions,
        "normal": normal_transactions,
        "fraud_rate": fraud_rate
    }

    return summary


# ===============================
# DATASET COLUMNS
# ===============================

def get_dataset_columns():
    """
    Lấy danh sách cột dataset
    """

    df = load_dataset()

    return list(df.columns)


# ===============================
# DATA FOR DBSCAN
# ===============================

def get_dbscan_data():
    """
    Lấy dữ liệu cho DBSCAN
    """

    df = load_dataset()

    X_scaled, _ = preprocess_pipeline(DATA_PATH)

    return df, X_scaled


# ===============================
# DATA FOR APRIORI
# ===============================

def get_apriori_data():
    """
    Chuẩn bị dữ liệu cho Apriori
    """

    df = load_dataset()

    # Apriori thường dùng dữ liệu nhị phân
    # Ví dụ chuyển Amount thành High/Low

    df_copy = df.copy()

    median_amount = df_copy["Amount"].median()

    df_copy["High_Amount"] = df_copy["Amount"] > median_amount

    df_copy["Fraud"] = df_copy[TARGET_COLUMN] == 1

    apriori_df = df_copy[["High_Amount", "Fraud"]]

    apriori_df = apriori_df.astype(bool)

    return apriori_df