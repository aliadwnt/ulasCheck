import pandas as pd
import re
import joblib
from collections import defaultdict

# === Load Pickle ===
vectorizer = joblib.load("model/vectorizer.pkl")  # Memuat vectorizer yang sudah dilatih
model = joblib.load("model/svm_model.pkl")  # Memuat model SVM yang sudah dilatih

# === Stopwords & Cleaning ===
from nltk.corpus import stopwords
stop_words = stopwords.words('indonesian')

# Fungsi untuk membersihkan teks (preprocessing)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Mengubah teks menjadi huruf kecil
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Menghapus karakter non-huruf
    text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi berlebih
    return text

# Fungsi untuk analisis aspek positif berdasarkan kata kunci
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
        if row[label_col] == 1:  # Menghitung hanya review positif
            for aspek, keywords in aspek_keywords.items():
                if any(kw in row[review_col] for kw in keywords):
                    aspek_counter[aspek] += 1
    return aspek_counter
# Fungsi utama untuk analisis data
def analyze_data(df):
    # Hapus baris dengan nilai null pada kolom 'Review' dan 'Rating'
    df.dropna(subset=["Review", "Rating"], inplace=True)
    
    # Lakukan pembersihan teks pada kolom 'Review'
    df["CleanReview"] = df["Review"].apply(clean_text)
    
    # Transformasi teks menjadi fitur menggunakan vectorizer yang telah dilatih
    X = vectorizer.transform(df["CleanReview"])  # Menggunakan vectorizer untuk transformasi teks
    
    # Prediksi label menggunakan model SVM yang telah dilatih dan simpan hasilnya di kolom 'PredictedLabel'
    df["PredictedLabel"] = model.predict(X)

    # Pastikan kolom 'PredictedLabel' ada setelah prediksi
    if 'PredictedLabel' not in df.columns:
        raise ValueError("Kolom 'PredictedLabel' tidak ditemukan setelah prediksi.")

    # Menghitung total review positif dan negatif berdasarkan rating (bukan prediksi model)
    total_pos = (df["Rating"] >= 4).sum()  # Menghitung review dengan rating >= 4 sebagai positif
    total_neg = (df["Rating"] < 4).sum()   # Menghitung review dengan rating < 4 sebagai negatif
    total_all = total_pos + total_neg
    persen_pos = round((total_pos / total_all) * 100, 2) if total_all > 0 else 0
    label_toko = "Direkomendasikan" if persen_pos >= 60 else "Tidak Direkomendasikan"

    # Melakukan analisis aspek positif
    aspek_result = analisis_aspek_positif(df)
    aspek_tertinggi = max(aspek_result, key=aspek_result.get, default="-")
    persen_tertinggi = round((aspek_result.get(aspek_tertinggi, 0) / total_all) * 100, 2) if total_all > 0 else 0

    # Mengembalikan hasil analisis
    return {
        "total": total_all,
        "positif": total_pos,
        "negatif": total_neg,
        "persen_pos": persen_pos,
        "label_toko": label_toko,
        "aspek_tertinggi": aspek_tertinggi,
        "persen_aspek": persen_tertinggi
    }
