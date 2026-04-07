# services/rule_service.py

"""
Rule Service
Quản lý việc chạy Apriori và xử lý luật kết hợp:
- Tạo frequent itemsets
- Tạo association rules
- Sắp xếp luật
- Lấy top luật để hiển thị dashboard
"""

import pandas as pd

from config import (
    MIN_SUPPORT,
    MIN_CONFIDENCE,
    MIN_LIFT,
    TOP_RULES_CHART
)

from models.apriori_model import apriori_pipeline

from services.data_service import get_apriori_data


# ===============================
# RUN APRIORI
# ===============================

def run_apriori():
    """
    Chạy thuật toán Apriori
    """

    apriori_data = get_apriori_data()

    rules = apriori_pipeline(apriori_data)

    return rules


# ===============================
# SORT RULES
# ===============================

def sort_rules_by_confidence(rules):
    """
    Sắp xếp luật theo confidence giảm dần
    """

    sorted_rules = rules.sort_values(
        by="confidence",
        ascending=False
    )

    return sorted_rules


# ===============================
# FILTER RULES
# ===============================

def filter_rules(rules):
    """
    Lọc luật theo điều kiện lift
    """

    filtered = rules[rules["lift"] >= MIN_LIFT]

    return filtered


# ===============================
# FORMAT RULES
# ===============================

def format_rules(rules):
    """
    Chuyển luật sang dạng dễ đọc
    """

    formatted = rules.copy()

    if "rule" not in formatted.columns:

        formatted["rule"] = (
            formatted["antecedents"].astype(str)
            + " → "
            + formatted["consequents"].astype(str)
        )

    return formatted


# ===============================
# GET TOP RULES
# ===============================

def get_top_rules(rules):
    """
    Lấy top rules cho biểu đồ
    """

    top_rules = rules.head(TOP_RULES_CHART)

    return top_rules


# ===============================
# RULE SUMMARY
# ===============================

def get_rule_summary(rules):
    """
    Tạo bảng thống kê luật
    """

    summary = pd.DataFrame({
        "rule": rules["rule"],
        "support": rules["support"],
        "confidence": rules["confidence"],
        "lift": rules["lift"]
    })

    return summary


# ===============================
# FULL RULE PIPELINE
# ===============================

def rule_analysis():
    """
    Pipeline hoàn chỉnh cho Apriori
    """

    rules = run_apriori()

    rules = filter_rules(rules)

    rules = sort_rules_by_confidence(rules)

    rules = format_rules(rules)

    top_rules = get_top_rules(rules)

    summary = get_rule_summary(rules)

    result = {
        "rules": rules,
        "top_rules": top_rules,
        "summary": summary
    }

    return result