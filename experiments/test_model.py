import os
import sys
import pandas as pd

# thêm path project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.fraud_detection import run_fraud_detection


def test_with_dataset_sizes(file_path, sizes):
    """
    Test hệ thống với nhiều kích thước dataset khác nhau
    """

    df = pd.read_csv(file_path)

    results = []

    print("\n==============================================")
    print(" CREDIT CARD FRAUD DETECTION - EXPERIMENT TEST")
    print("==============================================")

    for size in sizes:

        print(f"\nTesting dataset size: {size}")

        df_subset = df.head(size)

        temp_file = "temp_test.csv"
        df_subset.to_csv(temp_file, index=False)

        try:
            df_result, fraud_transactions, stats = run_fraud_detection(temp_file)

            result = {
                "Dataset Size": size,
                "Fraud Detected": stats["fraud_transactions"],
                "Fraud Rate (%)": stats["fraud_rate"]
            }

            results.append(result)

            print("Fraud Detected :", stats["fraud_transactions"])
            print("Fraud Rate     :", stats["fraud_rate"], "%")

        except Exception as e:
            print("ERROR:", e)

        if os.path.exists(temp_file):
            os.remove(temp_file)

    results_df = pd.DataFrame(results)

    return results_df


def main():

    file_path = "data/creditcard.csv"

    sizes = [
        1000,
        5000,
        10000,
        30000,
        50000
    ]

    results_df = test_with_dataset_sizes(file_path, sizes)

    print("\n\n============= FINAL RESULT TABLE =============")
    print(results_df.to_string(index=False))
    print("==============================================")

    os.makedirs("experiments", exist_ok=True)

    results_df.to_csv("experiments/results.csv", index=False)

    print("\nResults saved to: experiments/results.csv")


if __name__ == "__main__":
    main()