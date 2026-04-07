from models.fpgrowth_model import FPGrowthModel
import pandas as pd

def fpgrowth_pipeline(data, min_support=0.01, min_confidence=0.5):
    """
    Pipeline hoàn chỉnh cho FP-Growth
    """
    model = FPGrowthModel(min_support=min_support, min_confidence=min_confidence)
    model.fit(data)
    rules = model.generate_rules()
    # Định dạng lại rules cho giống Apriori
    if rules is not None and not rules.empty:
        rules['rule'] = (
            rules['antecedents'].apply(lambda x: ', '.join(list(x))) +
            ' → ' +
            rules['consequents'].apply(lambda x: ', '.join(list(x)))
        )
        rules = rules[['rule', 'support', 'confidence', 'lift']]
        rules['support'] = rules['support'].round(4)
        rules['confidence'] = rules['confidence'].round(4)
        rules['lift'] = rules['lift'].round(4)
    return rules
