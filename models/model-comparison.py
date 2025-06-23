import pandas as pd
import re
import numpy as np
import pickle, os, time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('indonesian')

# ====================
# 1. Preprocessing
# ====================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ====================
# 2. Load Dataset
# ====================
df = pd.read_csv("scraping-result/dataset.csv")
df.dropna(subset=["Review", "Rating"], inplace=True)
df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)
df["CleanReview"] = df["Review"].apply(clean_text)

# ====================
# 3. TF-IDF & Train-Test Split
# ====================
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df["CleanReview"])
y = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====================
# 4. Train Logistic Regression
# ====================
start_log = time.time()
log_model = LogisticRegression(class_weight='balanced', max_iter=1000)
log_model.fit(X_train, y_train)
end_log = time.time()

y_pred_log = log_model.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)
cm_log = confusion_matrix(y_test, y_pred_log)
time_log = round(end_log - start_log, 4)

# ====================
# 5. Train SVM
# ====================
start_svm = time.time()
svm_model = SVC(kernel='linear', class_weight='balanced')
svm_model.fit(X_train, y_train)
end_svm = time.time()

y_pred_svm = svm_model.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm)
time_svm = round(end_svm - start_svm, 4)

# ====================
# 6. Pilih Model Terbaik
# ====================
if acc_log >= acc_svm:
    best_model = log_model
    best_model_name = "Logistic Regression"
else:
    best_model = svm_model
    best_model_name = "SVM"

# ====================
# 7. Prediksi Semua Data
# ====================
df["PredictedLabel"] = best_model.predict(X)

def analisis_aspek_positif(df, review_col='CleanReview', label_col='PredictedLabel'):
    aspek_keywords = {
        "pengiriman": ["pengiriman", "kirim", "sampai", "kurir", "antar", "cepat", "tepat waktu", "on time", "kilat"],
        "pelayanan": ["pelayanan", "respon cepat", "layanan", "cs", "ramah", "sopan", "penjual baik", "admin ramah"],
        "produk": ["produk", "barang", "kualitas bagus", "bagus", "asli", "ori", "original", "sesuai deskripsi"],
        "harga": ["harga", "murah", "diskon", "promo", "worth it", "terjangkau", "value"],
        "packing": ["packing", "kemasan", "rapi", "aman", "bubble wrap", "dibungkus rapi"]
    }

    aspek_counter = defaultdict(int)
    for _, row in df.iterrows():
        if row[label_col] == 1:
            review = row[review_col]
            for aspek, keywords in aspek_keywords.items():
                if any(kw in review for kw in keywords):
                    aspek_counter[aspek] += 1
    return aspek_counter

aspek_result = analisis_aspek_positif(df)

# ====================
# 8. Evaluasi Toko
# ====================
total_pos = (df["PredictedLabel"] == 1).sum()
total_neg = (df["PredictedLabel"] == 0).sum()
total_all = total_pos + total_neg
persen_pos = round((total_pos / total_all) * 100, 2)
toko_label = "Direkomendasikan" if persen_pos >= 60 else "Tidak Direkomendasikan"
aspek_tertinggi = max(aspek_result, key=aspek_result.get)
jumlah_tertinggi = aspek_result[aspek_tertinggi]
persen_tertinggi = round((jumlah_tertinggi / total_all) * 100, 2)

# ====================
# 9. Simpan Evaluasi & Model
# ====================
evaluation_result = {
    "accuracy": {
        "Logistic Regression": round(acc_log * 100, 2),
        "SVM": round(acc_svm * 100, 2)
    },
    "time_process": {
        "Logistic Regression": time_log,
        "SVM": time_svm
    },
    "confusion_matrix": {
        "Logistic Regression": cm_log.tolist(),
        "SVM": cm_svm.tolist()
    },
    "best_model": best_model_name,
    "label_toko": toko_label,
    "persen_positif": persen_pos,
    "aspek_tertinggi": aspek_tertinggi,
    "jumlah_aspek": jumlah_tertinggi,
    "persen_aspek": persen_tertinggi,
    "aspek_result": dict(aspek_result)
}

os.makedirs("models", exist_ok=True)
with open("models/evaluation_result.pkl", "wb") as f:
    pickle.dump(evaluation_result, f)

with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print(">>> Evaluasi selesai dan disimpan ke folder 'models'") 