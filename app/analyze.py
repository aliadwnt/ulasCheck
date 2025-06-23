import pandas as pd
import re
import joblib
from collections import defaultdict
from nltk.corpus import stopwords

# === Load Pickle ===
vectorizer = joblib.load("model/vectorizer.pkl")  # Load vectorizer yang sudah dilatih
model = joblib.load("model/svm_model.pkl")        # Load model SVM yang sudah dilatih

# Stopwords bahasa Indonesia
stop_words = stopwords.words('indonesian')

# === Fungsi pembersihan teks ===
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()  # ke huruf kecil
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # hapus karakter selain huruf
    text = re.sub(r'\s+', ' ', text).strip()  # hapus spasi ganda
    return text

# === Analisis aspek berdasarkan kata kunci positif ===
def analisis_aspek_positif(df, review_col='CleanReview', label_col='PredictedLabel'):
    aspek_keywords = {
        "pengiriman": ["pengiriman", "kirim", "sampai", "cepat", "kurir", "antar", "lambat"],
        "pelayanan": ["pelayanan", "layanan", "respon", "ramah", "cs", "admin", "tanggap"],
        "produk": ["produk", "barang", "kualitas", "bagus", "baik", "ori", "asli"],
        "harga": ["harga", "murah", "diskon", "promo", "terjangkau", "value"],
        "packing": ["packing", "kemasan", "rapi", "aman", "bungkus", "bubble"]
    }

    aspek_counter = defaultdict(int)

    for _, row in df.iterrows():
        if row[label_col] == 1:  # Hanya ulasan positif
            for aspek, keywords in aspek_keywords.items():
                if any(kw in row[review_col] for kw in keywords):
                    aspek_counter[aspek] += 1
    return aspek_counter

# === Fungsi utama untuk analisis data ===
def analyze_data(df):
    # Hapus baris kosong
    df.dropna(subset=["Review", "Rating"], inplace=True)

    # Bersihkan teks
    df["CleanReview"] = df["Review"].apply(clean_text)

    # Transformasi teks
    X = vectorizer.transform(df["CleanReview"])

    # Prediksi label
    df["PredictedLabel"] = model.predict(X)

    # Hitung statistik
    total_pos = (df["Rating"] >= 4).sum()
    total_neg = (df["Rating"] < 4).sum()
    total_all = total_pos + total_neg
    persen_pos = round((total_pos / total_all) * 100, 2) if total_all > 0 else 0
    label_toko = "Direkomendasikan" if persen_pos >= 60 else "Tidak Direkomendasikan"

    # Analisis aspek
    aspek_result = analisis_aspek_positif(df)
    aspek_tertinggi = max(aspek_result, key=aspek_result.get, default="-")
    persen_tertinggi = round((aspek_result.get(aspek_tertinggi, 0) / total_all) * 100, 2) if total_all > 0 else 0

    # Hitung distribusi aspek dalam persentase
    total_aspek = sum(aspek_result.values())
    aspek_persen = {
        aspek: round((jumlah / total_aspek) * 100, 2)
        for aspek, jumlah in aspek_result.items()
    } if total_aspek > 0 else {}

    # Return semua data
    return {
        "total": total_all,
        "positif": total_pos,
        "negatif": total_neg,
        "persen_pos": persen_pos,
        "label_toko": label_toko,
        "aspek_tertinggi": aspek_tertinggi,
        "persen_aspek": persen_tertinggi,
        "aspek_persen": aspek_persen
    }
