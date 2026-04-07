import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

class FPGrowthModel:
    def __init__(self, min_support=0.01, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = None
        self.rules = None

    def fit(self, df, use_col=None):
        """
        Fit the FP-Growth model to the data.
        df: pandas DataFrame, one-hot encoded transaction data
        use_col: list of columns to use (optional)
        """
        if use_col:
            data = df[use_col]
        else:
            data = df
        self.frequent_itemsets = fpgrowth(data, min_support=self.min_support, use_colnames=True)
        return self.frequent_itemsets

    def generate_rules(self):
        if self.frequent_itemsets is None:
            raise ValueError("Run fit() before generating rules.")
        self.rules = association_rules(self.frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
        return self.rules

    def predict(self, df, use_col=None):
        """
        Predicts if a transaction is anomalous based on the absence of frequent patterns.
        Returns a list of booleans: True if likely fraud, False otherwise.
        """
        if self.frequent_itemsets is None:
            raise ValueError("Model not fitted yet.")
        if use_col:
            data = df[use_col]
        else:
            data = df
        predictions = []
        for _, row in data.iterrows():
            found = False
            for _, itemset in self.frequent_itemsets.iterrows():
                if set(itemset['itemsets']).issubset(set(row[row == 1].index)):
                    found = True
                    break
            predictions.append(not found)  # Not found = anomaly
        return predictions
