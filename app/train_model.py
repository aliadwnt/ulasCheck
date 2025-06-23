# train_model.py
import os
import re
import json
import time
import joblib
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from unidecode import unidecode
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

# === 1. Load Stopwords ===
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('indonesian')

# === 2. Cek file tuning parameter ===
tuning_path = "data/parameter-akurasi-svm_7525_Time_balance.xlsx"
if not os.path.exists(tuning_path):
    print(f"❌ File tuning tidak ditemukan: {tuning_path}")
    exit()
tuning_df = pd.read_excel(tuning_path)
idx_best = tuning_df["akurasi"].idxmax()
best_gamma = float(tuning_df.loc[idx_best, "gamma"])
best_C = float(tuning_df.loc[idx_best, "c"])
print(f"✅ Menggunakan gamma: {best_gamma}, C: {best_C}")

# === 3. Load Dataset ===
dataset_path = "dataset/Dataset.csv"
if not os.path.exists(dataset_path):
    print(f"❌ File dataset tidak ditemukan: {dataset_path}")
    exit()

df = pd.read_csv(dataset_path)
df.dropna(subset=["Review", "Rating"], inplace=True)
df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)

# === 4. Pembersihan Teks ===
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unidecode(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["CleanReview"] = df["Review"].apply(clean_text)

# === 5. TF-IDF ===
vectorizer = TfidfVectorizer(
    stop_words=stop_words,
    ngram_range=(1, 2),
    min_df=2,
    max_features=5000
)
X = vectorizer.fit_transform(df["CleanReview"])
y = df["Label"]

# === 6. Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 7. Training Model ===
model = SVC(kernel='rbf', gamma=best_gamma, C=best_C, class_weight='balanced', probability=True)
start_time = time.time()
model.fit(X_train, y_train)
training_duration = round(time.time() - start_time, 4)

# === 8. Evaluasi ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 === CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# === 9. Confusion Matrix ===
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"])
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.show()

# === 10. Analisis Aspek ===
df["PredictedLabel"] = model.predict(X)

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
        if row[label_col] == 1:
            review = row[review_col]
            for aspek, keywords in aspek_keywords.items():
                if any(kw in review for kw in keywords):
                    aspek_counter[aspek] += 1
    return aspek_counter

# Hitung total positif dan negatif sebelum aspek persen
total_pos = (df["PredictedLabel"] == 1).sum()
total_neg = (df["PredictedLabel"] == 0).sum()
total_all = total_pos + total_neg

aspek_result = analisis_aspek_positif(df)
aspek_persen = {
    aspek: round((jumlah / total_pos) * 100, 2) if total_pos > 0 else 0
    for aspek, jumlah in aspek_result.items()
}

# Dapatkan aspek terbanyak
aspek_tertinggi = max(aspek_result, key=aspek_result.get, default="-")
jumlah_tertinggi = aspek_result.get(aspek_tertinggi, 0)
persen_tertinggi = aspek_persen.get(aspek_tertinggi, 0)
persen_pos = round((total_pos / total_all) * 100, 2) if total_all > 0 else 0
toko_label = "Direkomendasikan" if persen_pos >= 60 else "Tidak Direkomendasikan"

# === 11. Ringkasan ===
print("\n📌 === PENILAIAN TOKO ===")
print(f"- Total Review: {total_all}")
print(f"- Positif: {total_pos} ({persen_pos}%)")
print(f"- Negatif: {total_neg} ({round(100 - persen_pos, 2)}%)")
print(f"- Label: {toko_label}")
print(f"- Aspek Positif Terbanyak: {aspek_tertinggi} ({jumlah_tertinggi} review, {persen_tertinggi}%)")
print("\n📈 Persentase Semua Aspek Positif:")
for aspek, persen in aspek_persen.items():
    print(f"- {aspek.title()}: {persen}%")

# === 12. Visualisasi Aspek ===
if aspek_persen:
    plt.figure(figsize=(10, 6))
    plt.bar(aspek_persen.keys(), aspek_persen.values(), color='green')
    plt.ylabel("Persentase dari Review Positif (%)")
    plt.title("Distribusi Aspek Positif")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("model/aspek_persen_positif.png")
    plt.show()

# === 13. Simpan Model dan Evaluasi ===
output_dir = "model"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(vectorizer, os.path.join(output_dir, "vectorizer.pkl"))
joblib.dump(model, os.path.join(output_dir, "svm_model.pkl"))
np.save(os.path.join(output_dir, "confusion_matrix.npy"), conf_matrix)

evaluation_summary = {
    "model_name": "SVM (RBF)",
    "gamma": float(best_gamma),
    "C": float(best_C),
    "accuracy": float(round(acc, 4)),
    "precision": float(round(prec, 4)),
    "recall": float(round(rec, 4)),
    "f1_score": float(round(f1, 4)),
    "training_time": float(training_duration),
    "total_review": int(total_all),
    "positif": int(total_pos),
    "negatif": int(total_neg),
    "persen_positif": float(persen_pos),
    "label_toko": toko_label,
    "aspek_tertinggi": aspek_tertinggi,
    "jumlah_aspek": int(jumlah_tertinggi),
    "persen_aspek": float(persen_tertinggi),
    "aspek_persen": {k: float(v) for k, v in aspek_persen.items()}
}

with open(os.path.join(output_dir, "evaluation_summary.json"), "w") as f:
    json.dump(evaluation_summary, f, indent=4)

print(f"\n✅ Model & evaluasi berhasil disimpan di folder: {output_dir}/")
