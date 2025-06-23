# app/services/analyze_service.py

import os
import re
import joblib
import numpy as np
import pandas as pd
from unidecode import unidecode
from collections import defaultdict

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unidecode(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_analysis(review_id):
    # Load model dan vectorizer
    model_path = "model/svm_model.pkl"
    vectorizer_path = "model/vectorizer.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("❌ Model atau vectorizer tidak ditemukan. Jalankan training terlebih dahulu.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Load dataset berdasarkan ID
    dataset_path = f"data/scraped/review_{review_id}.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ File review tidak ditemukan: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df.dropna(subset=["Review", "Rating"], inplace=True)
    df["CleanReview"] = df["Review"].apply(clean_text)

    # Prediksi menggunakan model pickle
    X = vectorizer.transform(df["CleanReview"])
    y_pred = model.predict(X)
    df["PredictedLabel"] = y_pred

    # Statistik prediksi
    total_pos = int((df["PredictedLabel"] == 1).sum())
    total_neg = int((df["PredictedLabel"] == 0).sum())
    total_all = total_pos + total_neg
    persen_pos = round((total_pos / total_all) * 100, 2) if total_all > 0 else 0
    toko_label = "Direkomendasikan" if persen_pos >= 60 else "Tidak Direkomendasikan"

    # Analisis aspek positif
    aspek_keywords = {
        "pengiriman": ["pengiriman", "kirim", "sampai", "cepat", "kurir", "antar", "lambat"],
        "pelayanan": ["pelayanan", "layanan", "respon", "ramah", "cs", "admin", "tanggap"],
        "produk": ["produk", "barang", "kualitas", "bagus", "baik", "ori", "asli"],
        "harga": ["harga", "murah", "diskon", "promo", "terjangkau", "value"],
        "packing": ["packing", "kemasan", "rapi", "aman", "bungkus", "bubble"]
    }

    aspek_counter = defaultdict(int)
    for _, row in df.iterrows():
        if row["PredictedLabel"] == 1:
            review = row["CleanReview"]
            for aspek, keywords in aspek_keywords.items():
                if any(kw in review for kw in keywords):
                    aspek_counter[aspek] += 1

    aspek_persen = {
        aspek: round((jumlah / total_pos) * 100, 2) if total_pos > 0 else 0
        for aspek, jumlah in aspek_counter.items()
    }
    aspek_tertinggi = max(aspek_counter, key=aspek_counter.get, default="-")
    persen_tertinggi = aspek_persen.get(aspek_tertinggi, 0)

    return {
        "total_all": total_all,
        "total_pos": total_pos,
        "total_neg": total_neg,
        "persen_pos": persen_pos,
        "toko_label": toko_label,
        "aspek": aspek_tertinggi,
        "persen_tertinggi": persen_tertinggi,
        "aspek_persen": aspek_persen
    }
