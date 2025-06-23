import os
import re
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from unidecode import unidecode
from nltk.corpus import stopwords
import nltk

# === 1. Unduh stopwords bahasa Indonesia ===
nltk.download('stopwords')
stop_words = stopwords.words('indonesian')

# === 2. Pastikan folder dataset dan data ada ===
dataset_path = "dataset/data_tuning.csv"
output_dir = "data"
output_file = os.path.join(output_dir, "parameter-akurasi-svm_7525_Time_balance.xlsx")
os.makedirs(output_dir, exist_ok=True)

# === 3. Load Dataset ===
if not os.path.exists(dataset_path):
    print(f"❌ File dataset tidak ditemukan: {dataset_path}")
    exit()

df = pd.read_csv(dataset_path)
df.dropna(subset=["Review", "Rating"], inplace=True)
df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)

# === 4. Preprocessing Teks ===
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

# === 5. TF-IDF Vectorizer ===
vectorizer = TfidfVectorizer(
    stop_words=stop_words,
    ngram_range=(1, 2),
    min_df=2,
    max_features=5000
)
X = vectorizer.fit_transform(df["CleanReview"]).toarray()
y = df["Label"].values

# === 6. Parameter Grid ===
C_range = np.arange(0.1, 3.1, 0.2)
gamma_range = np.arange(0.01, 1.1, 0.1)

# === 7. K-Fold Cross Validation ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

# === 8. Loop Tuning ===
total_combinations = len(C_range) * len(gamma_range)
combination_count = 1

for C in C_range:
    for gamma in gamma_range:
        accuracies = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            model = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced')
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            acc = accuracy_score(y[test_idx], y_pred)
            accuracies.append(acc)
        avg_acc = np.mean(accuracies)
        results.append({"c": C, "gamma": gamma, "akurasi": round(avg_acc, 4)})
        print(f"[{combination_count}/{total_combinations}] C={C}, gamma={gamma} → Akurasi: {round(avg_acc, 4)}")
        combination_count += 1

# === 9. Simpan ke Excel ===
result_df = pd.DataFrame(results)
result_df.to_excel(output_file, index=False)
print(f"\n✅ File hasil tuning berhasil disimpan ke: {output_file}")
