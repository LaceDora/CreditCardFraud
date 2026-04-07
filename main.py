from flask import Flask, render_template, redirect, url_for
import os
import uuid
import pandas as pd

# preprocessing
from preprocessing.preprocess import preprocess_all

# models
from models.dbscan_model import dbscan_pipeline
from utils.helpers import plot_clusters
from models.apriori_model import apriori_pipeline
from models.fpgrowth_pipeline import fpgrowth_pipeline
from models.kmeans_model import kmeans_pipeline

app = Flask(__name__)

# lưu kết quả phân tích
clustered_data = None
outliers = None
cluster_summary = None
rules_result = None
fpgrowth_result = None
kmeans_result = None
kmeans_summary = None
kmeans_centers = None
kmeans_inertia = None


# ==============================
# DASHBOARD
# ==============================

@app.route("/")
def dashboard():

    data, _, _ = preprocess_all()

    total = len(data)
    fraud = int(data["Class"].sum())
    normal = total - fraud

    summary = {
        "total_transactions": total,
        "fraud_transactions": fraud,
        "normal_transactions": normal
    }

    fraud_ratio = round(fraud / total, 4)

    top_transactions = data.sort_values(
        by="Amount",
        ascending=False
    ).head(10)

    return render_template(
        "index.html",
        summary=summary,
        fraud_ratio=fraud_ratio,
        top_transactions=top_transactions.to_dict(orient="records")
    )


# ==============================
# DATASET PAGE
# ==============================

@app.route("/dataset")
def dataset():

    data, _, _ = preprocess_all()

    summary = {
        "total_transactions": len(data),
        "fraud_transactions": int(data["Class"].sum()),
        "normal_transactions": int(len(data) - data["Class"].sum())
    }

    preview = data.head(100)

    return render_template(
        "dataset.html",
        summary=summary,
        dataset=preview.to_dict(orient="records"),
        column_count=data.shape[1],
        row_count=data.shape[0]
    )


# ==============================
# RUN ANALYSIS
# ==============================

@app.route("/run")
def run_analysis():

    global clustered_data
    global outliers
    global cluster_summary
    global rules_result
    global fpgrowth_result
    global kmeans_result
    global kmeans_summary
    global kmeans_centers
    global kmeans_inertia

    data, scaled_features, apriori_data = preprocess_all()

    # DBSCAN
    clustered_data, outliers, cluster_summary = dbscan_pipeline(
        data,
        scaled_features
    )

    # Apriori
    rules_result = apriori_pipeline(apriori_data)

    # FP-Growth
    fpgrowth_result = fpgrowth_pipeline(apriori_data)

    # KMeans
    kmeans_result, kmeans_summary, kmeans_centers, kmeans_inertia = kmeans_pipeline(
        data,
        scaled_features
    )

    # Đảm bảo luôn trả về response hợp lệ
    return redirect(url_for("clustering"))
# ASSOCIATION RULES PAGE
# ==============================

# FP-GROWTH RULES PAGE
# ==============================

@app.route("/fpgrowth")
def fpgrowth_rules():
    global fpgrowth_result
    if fpgrowth_result is None:
        return redirect(url_for("dashboard"))
    total_rules = len(fpgrowth_result)
    avg_support = round(fpgrowth_result["support"].mean(), 4)
    avg_confidence = round(fpgrowth_result["confidence"].mean(), 4)
    top_rules = fpgrowth_result.sort_values(
        by="lift",
        ascending=False
    ).head(5)
    # Sắp xếp: Nếu có cột 'Class', gian lận lên đầu
    fpgrowth_data = fpgrowth_result.copy()
    if 'Class' in fpgrowth_data.columns:
        fpgrowth_data = fpgrowth_data.sort_values(by="Class", ascending=False)
    return render_template(
        "rules.html",
        total_rules=total_rules,
        avg_support=avg_support,
        avg_confidence=avg_confidence,
        top_rules=top_rules.to_dict(orient="records"),
        rules=fpgrowth_data.to_dict(orient="records"),
        method="FP-Growth"
    )

    return redirect(url_for("clustering"))
# FP-GROWTH RULES PAGE
# ==============================


@app.route("/kmeans")
def kmeans_page():
    global kmeans_result, kmeans_summary, kmeans_centers, kmeans_inertia
    from flask import request
    if kmeans_result is None:
        return redirect(url_for("dashboard"))

    page = int(request.args.get("page", 1))
    per_page = 50
    selected_cluster = request.args.get("cluster", "all")
    selected_class = request.args.get("class", "all")

    df = kmeans_result.copy()
    # Lọc theo cụm
    if selected_cluster != "all":
        try:
            cluster_val = int(selected_cluster)
            df = df[df["KMeans_Cluster"] == cluster_val]
        except:
            pass
    # Lọc theo gian lận
    if selected_class != "all":
        if selected_class == "fraud":
            df = df[df["Class"] == 1]
        elif selected_class == "normal":
            df = df[df["Class"] == 0]

    # Sắp xếp gian lận lên đầu
    df = df.sort_values(by="Class", ascending=False)

    total_points = len(df)
    total_pages = (total_points + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    preview = df.iloc[start:end]

    cluster_counts = kmeans_summary.to_dict()
    cluster_list = sorted([int(c) for c in cluster_counts.keys()])

    # Vẽ và lưu biểu đồ KMeans (dùng lại plot_clusters, đổi tên cột cho phù hợp)
    kmeans_plot_path = None
    try:
        temp_data = kmeans_result.copy()
        if "KMeans_Cluster" in temp_data.columns:
            temp_data = temp_data.rename(columns={"KMeans_Cluster": "Cluster"})
        plot = plot_clusters(temp_data)
        if plot:
            filename = f"kmeans_{uuid.uuid4().hex}.png"
            save_path = os.path.join("static", "plots", filename)
            plot.savefig(save_path, bbox_inches="tight")
            plot.close()
            kmeans_plot_path = url_for("static", filename=f"plots/{filename}")
    except Exception as e:
        kmeans_plot_path = None

    return render_template(
        "clustering_kmeans.html",
        total_points=total_points,
        total_clusters=len(kmeans_summary),
        outliers=0,
        cluster_counts=cluster_counts,
        clustered_data=preview.to_dict(orient="records"),
        cluster_plot=kmeans_plot_path,
        centers=kmeans_centers,
        inertia=kmeans_inertia,
        page=page,
        total_pages=total_pages,
        selected_cluster=selected_cluster,
        selected_class=selected_class,
        cluster_list=cluster_list
    )


# ==============================
# CLUSTERING PAGE
# ==============================


from flask import request  # Đảm bảo đã import request

@app.route("/clustering")
def clustering():
    global clustered_data
    if clustered_data is None:
        return redirect(url_for("dashboard"))

    # Lấy các giá trị filter từ query string
    page = int(request.args.get("page", 1))
    per_page = 50
    selected_cluster = request.args.get("cluster", "all")
    selected_class = request.args.get("class", "all")

    df = clustered_data.copy()
    # Lọc theo cụm
    if selected_cluster != "all":
        try:
            cluster_val = int(selected_cluster)
            df = df[df["Cluster"] == cluster_val]
        except:
            if selected_cluster == "outlier":
                df = df[df["Cluster"] == -1]
    # Lọc theo gian lận
    if selected_class != "all":
        if selected_class == "fraud":
            df = df[df["Class"] == 1]
        elif selected_class == "normal":
            df = df[df["Class"] == 0]

    # Sắp xếp gian lận lên đầu
    df = df.sort_values(by="Class", ascending=False)

    total_points = len(df)
    total_pages = (total_points + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    preview = df.iloc[start:end]

    cluster_counts = clustered_data["Cluster"].value_counts().to_dict()
    total_clusters = len([c for c in cluster_counts.keys() if c != -1])
    outlier_count = cluster_counts.get(-1, 0)

    # Lấy danh sách các cụm cho dropdown
    cluster_list = sorted([c for c in cluster_counts.keys() if c != -1])
    if outlier_count > 0:
        cluster_list.append("outlier")

    # Vẽ và lưu biểu đồ DBSCAN
    dbscan_plot_path = None
    try:
        temp_data = clustered_data.copy()
        plot = plot_clusters(temp_data)
        if plot:
            filename = f"dbscan_{uuid.uuid4().hex}.png"
            save_path = os.path.join("static", "plots", filename)
            plot.savefig(save_path, bbox_inches="tight")
            plot.close()
            dbscan_plot_path = url_for("static", filename=f"plots/{filename}")
    except Exception as e:
        dbscan_plot_path = None

    return render_template(
        "clustering.html",
        total_points=total_points,
        total_clusters=total_clusters,
        outliers=outlier_count,
        cluster_counts=cluster_counts,
        clustered_data=preview.to_dict(orient="records"),
        cluster_plot=dbscan_plot_path,
        page=page,
        total_pages=total_pages,
        selected_cluster=selected_cluster,
        selected_class=selected_class,
        cluster_list=cluster_list
    )


# ==============================
# ASSOCIATION RULES PAGE
# ==============================

@app.route("/rules")
def rules():
    if rules_result is None:
        return redirect(url_for("dashboard"))

    total_rules = len(rules_result)
    avg_support = round(rules_result["support"].mean(), 4)
    avg_confidence = round(rules_result["confidence"].mean(), 4)
    top_rules = rules_result.sort_values(
        by="lift",
        ascending=False
    ).head(5)

    # Sắp xếp: Nếu có cột 'Class', gian lận lên đầu
    rules_data = rules_result.copy()
    if 'Class' in rules_data.columns:
        rules_data = rules_data.sort_values(by="Class", ascending=False)
    return render_template(
        "rules.html",
        total_rules=total_rules,
        avg_support=avg_support,
        avg_confidence=avg_confidence,
        top_rules=top_rules.to_dict(orient="records"),
        rules=rules_data.to_dict(orient="records")
    )

    return render_template(
        "clustering.html",
        total_points=total_points,
        total_clusters=total_clusters,
        outliers=outlier_count,
        cluster_counts=cluster_counts_dict,
        clustered_data=preview.to_dict(orient="records"),
        cluster_plot=dbscan_plot_path
    )
# RUN SERVER
# ==============================

if __name__ == "__main__":
    app.run(debug=True)