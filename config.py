# config.py

# File cấu hình hệ thống Fraud Detection

import os

# ===============================

# PROJECT PATH

# ===============================

# Thư mục gốc của project

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================

# DATASET CONFIG

# ===============================

# Thư mục chứa dữ liệu

DATA_FOLDER = os.path.join(BASE_DIR, "data")

# File dataset credit card

DATA_PATH = os.path.join(DATA_FOLDER, "creditcard.csv")

# Số dòng đọc khi demo (giảm để chạy nhanh)

MAX_ROWS = 50000

# ===============================

# DATASET COLUMN CONFIG

# ===============================

# Cột label gian lận

TARGET_COLUMN = "Class"

# Cột số tiền giao dịch

AMOUNT_COLUMN = "Amount"

# Cột thời gian

TIME_COLUMN = "Time"

# ===============================

# DBSCAN PARAMETERS

# ===============================

# Bán kính lân cận

DBSCAN_EPS = 0.5

# Số điểm tối thiểu để tạo cluster

DBSCAN_MIN_SAMPLES = 5

# Nhãn outlier

OUTLIER_LABEL = -1

# ===============================

# APRIORI PARAMETERS

# ===============================

# Support tối thiểu

MIN_SUPPORT = 0.01

# Confidence tối thiểu

MIN_CONFIDENCE = 0.5

# Lift tối thiểu

MIN_LIFT = 1.0

# ===============================

# UI CONFIG

# ===============================

# Số dòng hiển thị trong bảng dataset

DATASET_PREVIEW_ROWS = 100

# Số dòng hiển thị outlier

OUTLIER_PREVIEW_ROWS = 50

# Số luật hiển thị trên biểu đồ

TOP_RULES_CHART = 10

# ===============================

# SYSTEM CONFIG

# ===============================

# Chế độ debug Flask

DEBUG_MODE = True

# Port chạy Flask

PORT = 5000
