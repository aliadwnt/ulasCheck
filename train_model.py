import os, re, joblib, nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

# === Download stopwords jika belum ada ===
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('indonesian')

# === Fungsi pembersihan teks ===
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === Load dataset ===
dataset_path = "dataset/Dataset.csv"
if not os.path.exists(dataset_path):
    print(f"❌ File dataset tidak ditemukan di path: {dataset_path}")
    exit()

df = pd.read_csv(dataset_path)
df.dropna(subset=["Review", "Rating"], inplace=True)
df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)
df["CleanReview"] = df["Review"].apply(clean_text)

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df["CleanReview"])
y = df["Label"]

# === Split data untuk training & testing ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Training SVM Model ===
model = SVC(kernel='linear', class_weight='balanced', probability=True)
model.fit(X_train, y_train)

# === Evaluasi model ===
y_pred = model.predict(X_test)
print("\n📊 === CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# === Confusion Matrix ===
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"])
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# === Prediksi ulang seluruh data untuk analisis aspek ===
df["PredictedLabel"] = model.predict(X)

# === Fungsi analisis aspek positif ===
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

# === Proses penilaian dan analisis aspek ===
aspek_result = analisis_aspek_positif(df)
aspek_tertinggi = max(aspek_result, key=aspek_result.get, default="-")
jumlah_tertinggi = aspek_result.get(aspek_tertinggi, 0)

total_pos = (df["Label"] == 1).sum()
total_neg = (df["Label"] == 0).sum()
total_all = total_pos + total_neg
persen_pos = round((total_pos / total_all) * 100, 2) if total_all > 0 else 0
persen_tertinggi = round((jumlah_tertinggi / total_all) * 100, 2) if total_all > 0 else 0
toko_label = "Direkomendasikan" if persen_pos >= 60 else "Tidak Direkomendasikan"

# === Tampilkan Ringkasan Penilaian ===
print("\n📌 === PENILAIAN TOKO ===")
print(f"- Total Review: {total_all}")
print(f"- Positif: {total_pos} ({persen_pos}%)")
print(f"- Negatif: {total_neg} ({round(100 - persen_pos, 2)}%)")
print(f"- Label: {toko_label}")
print(f"- Aspek Positif Terbanyak: {aspek_tertinggi} ({jumlah_tertinggi} review, {persen_tertinggi}%)")

# === Visualisasi aspek positif ===
if aspek_result:
    plt.figure(figsize=(10, 6))
    plt.bar(aspek_result.keys(), aspek_result.values(), color='green')
    plt.ylabel("Jumlah Review Positif")
    plt.title("Aspek Positif yang Paling Sering Disebutkan")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

# === Simpan model & vectorizer ke file .pkl ===
output_dir = "model"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(vectorizer, os.path.join(output_dir, "vectorizer.pkl"))
joblib.dump(model, os.path.join(output_dir, "svm_model.pkl"))
print(f"\n✅ Model dan Vectorizer berhasil disimpan di folder: {output_dir}/")
