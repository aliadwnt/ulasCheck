# from flask import Flask, render_template, request, redirect, url_for, flash
# from scraping.scrapingAll import shopee
# from analysis.analyze import analyze_data
# from flask import request

# app = Flask(__name__, template_folder="templates")

# @app.route("/", methods=["GET", "POST"])
# def dashboard_public():
#     result = None
#     data_preview = None

#     if request.method == "POST":
#         link = request.form.get("link")
#         if link:
#             cookies_path = "cookies.json"
#             df = shopee(link, cookies_path)  # Pastikan shopee() return DataFrame

#             if df is not None and not df.empty:
#                 # Simpan ke CSV
#                 csv_path = "scraping-result/datasetNew.csv"
#                 df.to_csv(csv_path, index=False)

#                 # Analisis (pastikan analyze_data pakai datasetNew.csv)
#                 result = analyze_data(csv_path)

#                 # Tampilkan 5 ulasan pertama
#                 data_preview = df.head(5).to_dict(orient="records")
#             else:
#                 flash("Gagal mengambil data dari toko Shopee. Periksa link dan cookie.", "error")

#     return render_template("pages/public/dashboard.html", result=result, data_preview=data_preview)

# @app.route("/about-us")
# def aboutUs():
#     return render_template("pages/public/about-us.html")

# @app.route("/login")
# def login():
#     return render_template("pages/login.html")

# @app.route("/admin/dashboard")
# def adminDashboard():
#     return render_template("pages/admin/dashboard.html")

# @app.route("/admin/dataset")
# def adminDataset():
#     return render_template("pages/admin/dataset.html")

# @app.route("/admin/evaluation")
# def adminEvaluation():
#     return render_template("pages/admin/evaluation.html")

# @app.route("/admin/hitsory")
# def adminHistory():
#     return render_template("pages/admin/history.html")

# # @app.route("/admin/evaluasi")
# # def admin_evaluasi():
# #     return render_template("pages/admin/evaluasi.html")

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, render_template, request, flash
from scraping.scrapingAll import shopee
import pandas as pd
import joblib
import re
from collections import defaultdict
from datetime import datetime

app = Flask(__name__, template_folder="templates")
app.secret_key = 'your_secret_key'  # Dibutuhkan untuk flash()

# Load pipeline hanya sekali saat server start
pipeline = joblib.load("model/pipeline.pkl")

# Preprocessing text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Analisis hasil prediksi
def analyze_data(csv_path):
    df = pd.read_csv(csv_path)
    df["CleanReview"] = df["Review"].apply(clean_text)
    df["PredictedLabel"] = pipeline.predict(df["CleanReview"])

    total_pos = (df["PredictedLabel"] == 1).sum()
    total_neg = (df["PredictedLabel"] == 0).sum()
    total_all = total_pos + total_neg
    persen_pos = round((total_pos / total_all) * 100, 2) if total_all > 0 else 0
    label_toko = "Direkomendasikan" if persen_pos >= 60 else "Tidak Direkomendasikan"

    aspek_keywords = {
        "produk": ["bagus", "kualitas", "ori", "mantap"],
        "harga": ["murah", "harga", "worth"],
        "pengiriman": ["cepat", "sampai", "kurir"],
        "pelayanan": ["ramah", "respon", "admin"],
        "packing": ["rapi", "aman", "bubble"]
    }

    aspek_counter = defaultdict(int)
    for _, row in df.iterrows():
        if row["PredictedLabel"] == 1:
            for aspek, keywords in aspek_keywords.items():
                if any(kw in row["CleanReview"] for kw in keywords):
                    aspek_counter[aspek] += 1

    aspek_tertinggi = max(aspek_counter, key=aspek_counter.get, default="-")
    jumlah_tertinggi = aspek_counter.get(aspek_tertinggi, 0)
    persen_tertinggi = round((jumlah_tertinggi / total_all) * 100, 2) if total_all > 0 else 0

    return {
        "total": total_all,
        "positif": total_pos,
        "negatif": total_neg,
        "persen_positif": persen_pos,
        "label_toko": label_toko,
        "aspek_tertinggi": aspek_tertinggi,
        "jumlah_aspek": jumlah_tertinggi,
        "persen_aspek": persen_tertinggi
    }

@app.context_processor
def inject_now():
    return {'now': datetime.now()}

@app.route("/", methods=["GET", "POST"])
def dashboard_public():
    result = None
    data_preview = None

    if request.method == "POST":
        link = request.form.get("link")
        if link:
            cookies_path = "cookies.json"
            df = shopee(link, cookies_path)

            if df is not None and not df.empty:
                csv_path = "scraping-result/datasetNew.csv"
                df.to_csv(csv_path, index=False)

                result = analyze_data(csv_path)
                data_preview = df.head(5).to_dict(orient="records")
            else:
                flash("Gagal mengambil data dari toko Shopee. Periksa link dan cookie.", "error")

    return render_template("pages/public/dashboard.html", result=result, data_preview=data_preview)

@app.route("/about-us")
def aboutUs():
    return render_template("pages/public/about-us.html")

@app.route("/login")
def login():
    return render_template("pages/login.html")

@app.route("/admin/dashboard")
def adminDashboard():
    return render_template("pages/admin/dashboard.html")
# @app.route("/admin/dashboard")
# def adminDashboard():
#     return render_template("components/footer.html")

@app.route("/admin/dataset")
def adminDataset():
    return render_template("pages/admin/dataset.html")

@app.route("/admin/evaluation")
def adminEvaluation():
    return render_template("pages/admin/evaluation.html")

@app.route("/admin/history")
def adminHistory():
    return render_template("pages/admin/history.html")

if __name__ == "__main__":
    app.run(debug=True)
