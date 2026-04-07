import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data():
    """
    Đọc dữ liệu CSV
    """

    df = pd.read_csv("data/creditcard.csv", nrows=50000)

    return df


def split_features_labels(df):
    """
    Tách feature và label
    """

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y


def scale_features(X):
    """
    Chuẩn hóa dữ liệu
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled


def prepare_apriori_data(df):
    """
    Chuẩn bị dữ liệu cho Apriori
    """

    apriori_df = df.copy()

    # Phân loại Amount thành nhiều mức
    apriori_df["Amount_Low"] = apriori_df["Amount"] < 50
    apriori_df["Amount_Medium"] = (apriori_df["Amount"] >= 50) & (apriori_df["Amount"] < 200)
    apriori_df["Amount_High"] = apriori_df["Amount"] >= 200

    # Phân loại Time thành các khung giờ (giả sử Time là số giây, chia thành 4 khung)
    apriori_df["Time_Morning"] = (apriori_df["Time"] % 86400 < 21600)
    apriori_df["Time_Day"] = ((apriori_df["Time"] % 86400 >= 21600) & (apriori_df["Time"] % 86400 < 43200))
    apriori_df["Time_Evening"] = ((apriori_df["Time"] % 86400 >= 43200) & (apriori_df["Time"] % 86400 < 64800))
    apriori_df["Time_Night"] = (apriori_df["Time"] % 86400 >= 64800)

    # Đặc trưng gian lận
    apriori_df["Fraud"] = apriori_df["Class"] == 1

    # Chỉ lấy các cột one-hot
    apriori_df = apriori_df[[
        "Amount_Low", "Amount_Medium", "Amount_High",
        "Time_Morning", "Time_Day", "Time_Evening", "Time_Night",
        "Fraud"
    ]]

    # Đảm bảo kiểu bool
    apriori_df = apriori_df.astype(bool)

    return apriori_df


def preprocess_all():
    """
    Pipeline tiền xử lí hoàn chỉnh
    """

    df = load_data()

    X, y = split_features_labels(df)

    X_scaled = scale_features(X)

    apriori_data = prepare_apriori_data(df)

    return df, X_scaled, apriori_data