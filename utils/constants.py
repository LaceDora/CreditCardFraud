# utils/constants.py

"""
Constants
Các hằng số dùng chung trong hệ thống Fraud Detection
"""


# =================================
# DATASET COLUMNS
# =================================

# Cột nhãn gian lận
TARGET_COLUMN = "Class"

# Cột số tiền giao dịch
AMOUNT_COLUMN = "Amount"

# Cột thời gian giao dịch
TIME_COLUMN = "Time"

# Cột cluster sau khi chạy DBSCAN
CLUSTER_COLUMN = "Cluster"


# =================================
# FRAUD LABELS
# =================================

# Nhãn giao dịch bình thường
NORMAL_LABEL = 0

# Nhãn giao dịch gian lận
FRAUD_LABEL = 1

# Nhãn outlier trong DBSCAN
OUTLIER_LABEL = -1


# =================================
# UI DISPLAY
# =================================

# Số dòng preview dataset
DATASET_PREVIEW_ROWS = 100

# Số dòng hiển thị outlier
OUTLIER_PREVIEW_ROWS = 50

# Số luật hiển thị trong biểu đồ
TOP_RULES_CHART = 10

# Số giao dịch lớn nhất hiển thị
TOP_TRANSACTIONS = 10


# =================================
# CHART TITLES
# =================================

FRAUD_CHART_TITLE = "Tỷ lệ giao dịch gian lận"

CLUSTER_CHART_TITLE = "Phân bố cụm giao dịch"

RULES_CHART_TITLE = "Top luật phát hiện gian lận"


# =================================
# TABLE COLUMNS
# =================================

RULE_TABLE_COLUMNS = [
    "rule",
    "support",
    "confidence",
    "lift"
]


# =================================
# SYSTEM MESSAGES
# =================================

MSG_NO_DATA = "Không có dữ liệu"

MSG_NO_CLUSTER = "Chưa chạy clustering"

MSG_NO_RULES = "Chưa có luật Apriori"


# =================================
# COLOR SETTINGS (UI)
# =================================

COLOR_FRAUD = "#dc3545"

COLOR_NORMAL = "#28a745"

COLOR_CLUSTER = "#007bff"