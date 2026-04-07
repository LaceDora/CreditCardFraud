# models/apriori_model.py

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def generate_frequent_itemsets(data, min_support=0.001):
    """
    Tìm các tập phổ biến (frequent itemsets)
    """

    frequent_itemsets = apriori(
        data,
        min_support=min_support,
        use_colnames=True
    )

    return frequent_itemsets


def generate_rules(frequent_itemsets, min_confidence=0.1):
    """
    Tạo luật kết hợp từ các itemsets
    """

    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    return rules


def sort_rules(rules):
    """
    Sắp xếp luật theo confidence giảm dần
    """

    if rules.empty:
        return rules

    return rules.sort_values(by="confidence", ascending=False)


def format_rules(rules):
    """
    Chuyển luật sang dạng dễ đọc
    """

    if rules.empty:
        return pd.DataFrame(columns=["rule", "support", "confidence", "lift"])

    formatted = rules.copy()

    formatted["rule"] = (
        formatted["antecedents"].apply(lambda x: ', '.join(list(x)))
        + " → "
        + formatted["consequents"].apply(lambda x: ', '.join(list(x)))
    )

    formatted = formatted[["rule", "support", "confidence", "lift"]]

    # làm tròn số cho đẹp
    formatted["support"] = formatted["support"].round(4)
    formatted["confidence"] = formatted["confidence"].round(4)
    formatted["lift"] = formatted["lift"].round(4)

    return formatted


def apriori_pipeline(data):
    """
    Pipeline hoàn chỉnh cho Apriori
    """

    frequent_itemsets = generate_frequent_itemsets(data)

    rules = generate_rules(frequent_itemsets)

    rules = sort_rules(rules)

    rules = format_rules(rules)

    return rules
