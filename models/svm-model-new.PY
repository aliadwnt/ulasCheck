import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('indonesian')

# --- Preprocessing ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Load Data ---
df = pd.read_csv("scraping-result/Dataset.csv")
df.dropna(subset=["Review", "Rating"], inplace=True)
df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)
df["CleanReview"] = df["Review"].apply(clean_text)

# --- Vectorization ---
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df["CleanReview"])
y = df["Label"]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- SVM Training ---
model = SVC(kernel='linear', class_weight='balanced')
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
labels = ["Negatif (0)", "Positif (1)"]
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# --- Predict All Data ---
df["PredictedLabel"] = model.predict(X)

# --- Aspect Analysis ---
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

# --- Store Evaluation ---
total_pos = (df["PredictedLabel"] == 1).sum()
total_neg = (df["PredictedLabel"] == 0).sum()
total_all = total_pos + total_neg
persen_pos = round((total_pos / total_all) * 100, 2)
toko_label = "Direkomendasikan" if persen_pos >= 60 else "Tidak Direkomendasikan"
aspek_result = analisis_aspek_positif(df)
aspek_tertinggi = max(aspek_result, key=aspek_result.get)
jumlah_tertinggi = aspek_result[aspek_tertinggi]
persen_tertinggi = round((jumlah_tertinggi / total_all) * 100, 2)

# --- Summary ---
print("\n=== PENILAIAN TOKO ===")
print(f"- Total Ulasan: {total_all}")
print(f"- Positif: {total_pos} ({persen_pos}%)")
print(f"- Negatif: {total_neg} ({round((total_neg / total_all) * 100, 2)}%)")
print(f"- Label Toko: {toko_label}")
print(f"- Aspek yang menonjol (positif): '{aspek_tertinggi}' sebanyak {jumlah_tertinggi} review ({persen_tertinggi}%).")

# --- Visualization ---
labels = list(aspek_result.keys())
counts = [aspek_result[k] for k in labels]
plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color='green')
plt.ylabel("Jumlah Review Positif")
plt.title("Aspek Positif yang Paling Sering Disebutkan")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
