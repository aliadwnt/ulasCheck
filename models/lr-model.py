# import pandas as pd
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import defaultdict
# import nltk

# # Download stopwords
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = stopwords.words('indonesian')

# # ====================
# # 1. Preprocessing
# # ====================
# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # inisialisasi df
# df = pd.read_csv("scraping-result/Dataset.csv")

# # Load dataDataset.csv")
# df.dropna(subset=["Review", "Rating"], inplace=True)

# # Label: 1 jika rating >= 4, else 0
# df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)

# # Clean review text
# df["CleanReview"] = df["Review"].apply(clean_text)

# # ====================
# # 2. Vectorization
# # ====================
# vectorizer = TfidfVectorizer(stop_words=stop_words)
# X = vectorizer.fit_transform(df["CleanReview"])
# y = df["Label"]

# # ====================
# # 3. Model Training
# # ====================
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LogisticRegression(class_weight='balanced', max_iter=1000)
# model.fit(X_train, y_train)

# # Evaluation
# y_pred = model.predict(X_test)
# print("\n=== CLASSIFICATION REPORT ===")
# print(classification_report(y_test, y_pred))

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix Logistic Regression")
# plt.show()

# # ====================
# # 4. Analisis Aspek
# # ====================
# def analisis_aspek(df, label_col='Label', review_col='CleanReview'):
#     aspek_keywords = {
#     "pengiriman": ["pengiriman", "kirim", "pengantar", "kurir", "antar", "sampai", "cepat", "lama", "lambat", "resi", "delay", "terlambat"],
#     "pelayanan": ["pelayanan", "layanan", "cs", "customer service", "respon", "tanggapan", "ramah", "komplain", "balas", "sopan", "cepat"],
#     "produk": ["produk", "barang", "kualitas", "baik", "buruk", "rusak", "cacat", "fungsi", "bagus", "jelek", "asli", "palsu", "ori", "original", "wangi", "recommend"],
#     "harga": ["harga", "murah", "mahal", "diskon", "promo", "potongan", "worth it", "sesuai", "value", "terjangkau", "overprice"],
#     "packing": ["packing", "paket", "bungkus", "kemasan", "wrap", "bubble", "rapi", "aman", "dus", "kotak", "plastik"]}

#     aspek_counter = defaultdict(lambda: {'positif': 0, 'negatif': 0})

#     for i, row in df.iterrows():
#         review = row[review_col]
#         label = 'positif' if row[label_col] == 1 else 'negatif'
#         for aspek, keywords in aspek_keywords.items():
#             if any(kw in review for kw in keywords):
#                 aspek_counter[aspek][label] += 1

#     return aspek_counter

# # Tambah kolom prediksi untuk semua data (bukan hanya test set)
# df["PredictedLabel"] = model.predict(X)

# # Hitung total ulasan positif/negatif
# total_pos = (df["PredictedLabel"] == 1).sum()
# total_neg = (df["PredictedLabel"] == 0).sum()
# total_all = total_pos + total_neg
# persen_pos = round((total_pos / total_all) * 100, 2)
# persen_neg = round((total_neg / total_all) * 100, 2)

# print("\n=== RANGKUMAN ULASAN ===")
# print(f"- Total ulasan: {total_all}")
# print(f"- Ulasan Positif (Direkomendasikan): {total_pos} ({persen_pos}%)")
# print(f"- Ulasan Negatif (Tidak Direkomendasikan): {total_neg} ({persen_neg}%)")

# # Analisis aspek
# aspek_result = analisis_aspek(df, label_col="PredictedLabel")

# print("\n=== HASIL ANALISIS ASPEK ===")
# for aspek, nilai in aspek_result.items():
#     total = nilai['positif'] + nilai['negatif']
#     if total == 0:
#         continue
#     print(f"- {aspek.capitalize()}: {nilai['positif']} positif, {nilai['negatif']} negatif")

# # ====================
# # 5. Visualisasi
# # ====================
# labels = list(aspek_result.keys())
# positif = [aspek_result[a]['positif'] for a in labels]
# negatif = [aspek_result[a]['negatif'] for a in labels]

# x = range(len(labels))
# plt.figure(figsize=(10, 6))
# plt.bar(x, positif, label='Positif', color='green')
# plt.bar(x, negatif, bottom=positif, label='Negatif', color='red')
# plt.xticks(x, labels)
# plt.ylabel('Jumlah Ulasan')
# plt.title('Analisis Aspek Pelayanan dari Ulasan Toko')
# plt.legend()
# plt.tight_layout()
# plt.show()
# import pandas as pd
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import defaultdict
# import nltk

# # Download stopwords
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = stopwords.words('indonesian')

# # ====================
# # 1. Preprocessing
# # ====================
# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # ====================
# # 2. Load & Clean Dataset
# # ====================
# df = pd.read_csv("scraping-result/Dataset.csv")
# df.dropna(subset=["Review", "Rating"], inplace=True)
# df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)
# df["CleanReview"] = df["Review"].apply(clean_text)

# # ====================
# # 3. Vectorization & Model Training
# # ====================
# vectorizer = TfidfVectorizer(stop_words=stop_words)
# X = vectorizer.fit_transform(df["CleanReview"])
# y = df["Label"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LogisticRegression(class_weight='balanced', max_iter=1000)
# model.fit(X_train, y_train)

# # ====================
# # 4. Prediksi Semua Data
# # ====================
# df["PredictedLabel"] = model.predict(X)

# # ====================
# # 5. Analisis Aspek
# # ====================
# def analisis_aspek(df, label_col='PredictedLabel', review_col='CleanReview'):
#     aspek_keywords = {
#         "pengiriman": ["pengiriman", "kirim", "pengantar", "kurir", "antar", "sampai", "cepat", "lama", "lambat", "resi", "delay", "terlambat"],
#         "pelayanan": ["pelayanan", "layanan", "cs", "customer service", "respon", "tanggapan", "ramah", "komplain", "balas", "sopan", "cepat"],
#         "produk": ["produk", "barang", "kualitas", "baik", "buruk", "rusak", "cacat", "fungsi", "bagus", "jelek", "asli", "palsu", "ori", "original", "wangi", "recommend"],
#         "harga": ["harga", "murah", "mahal", "diskon", "promo", "potongan", "worth it", "sesuai", "value", "terjangkau", "overprice"],
#         "packing": ["packing", "paket", "bungkus", "kemasan", "wrap", "bubble", "rapi", "aman", "dus", "kotak", "plastik"]
#     }

#     aspek_counter = defaultdict(lambda: {'positif': 0, 'negatif': 0})
#     aspek_present = defaultdict(list)

#     for _, row in df.iterrows():
#         review = row[review_col]
#         label = 'positif' if row[label_col] == 1 else 'negatif'
#         for aspek, keywords in aspek_keywords.items():
#             if any(kw in review for kw in keywords):
#                 aspek_counter[aspek][label] += 1
#                 aspek_present[aspek].append(review)

#     return aspek_counter, aspek_present

# # Analisis aspek
# aspek_result, _ = analisis_aspek(df)

# # ====================
# # 6. Evaluasi Toko & Alasan Positif
# # ====================
# total_pos = (df["PredictedLabel"] == 1).sum()
# total_neg = (df["PredictedLabel"] == 0).sum()
# total_all = total_pos + total_neg
# persen_pos = round((total_pos / total_all) * 100, 2)

# # Label toko
# toko_label = "Direkomendasikan" if persen_pos >= 60 else "Tidak Direkomendasikan"

# # Cari aspek positif dominan
# aspek_positif_terbanyak = max(aspek_result.items(), key=lambda x: x[1]['positif'])[0]
# jumlah_positif_aspek = aspek_result[aspek_positif_terbanyak]['positif']
# persen_aspek = round((jumlah_positif_aspek / total_all) * 100, 2)

# # ====================
# # 7. Ringkasan
# # ====================
# print("\n=== PENILAIAN TOKO ===")
# print(f"- Total Ulasan: {total_all}")
# print(f"- Positif: {total_pos} ({persen_pos}%)")
# print(f"- Negatif: {total_neg} ({round((total_neg / total_all) * 100, 2)}%)")
# print(f"- Label Toko: {toko_label}")
# print(f"- Toko {toko_label.lower()}, Karena aspek '{aspek_positif_terbanyak}' memiliki kontribusi positif sebanyak {jumlah_positif_aspek} review ({persen_aspek}%).")

# # ====================
# # 8. Visualisasi Aspek
# # ====================
# labels = list(aspek_result.keys())
# positif = [aspek_result[a]['positif'] for a in labels]
# negatif = [aspek_result[a]['negatif'] for a in labels]

# x = range(len(labels))
# plt.figure(figsize=(10, 6))
# plt.bar(x, positif, label='Positif', color='green')
# plt.bar(x, negatif, bottom=positif, label='Negatif', color='red')
# plt.xticks(x, labels)
# plt.ylabel('Jumlah Ulasan')
# plt.title('Analisis Aspek dari Review Toko')
# plt.legend()
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import defaultdict
# import nltk

# # Download stopwords
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = stopwords.words('indonesian')

# # ====================
# # 1. Preprocessing
# # ====================
# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # ====================
# # 2. Load & Clean Dataset
# # ====================
# df = pd.read_csv("scraping-result/Dataset.csv")
# df.dropna(subset=["Review", "Rating"], inplace=True)
# df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)
# df["CleanReview"] = df["Review"].apply(clean_text)

# # ====================
# # 3. Vectorization & Model Training
# # ====================
# vectorizer = TfidfVectorizer(stop_words=stop_words)
# X = vectorizer.fit_transform(df["CleanReview"])
# y = df["Label"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LogisticRegression(class_weight='balanced', max_iter=1000)
# model.fit(X_train, y_train)

# # ====================
# # 4. Evaluation + Confusion Matrix
# # ====================
# y_pred = model.predict(X_test)
# print("\n=== CLASSIFICATION REPORT ===")
# print(classification_report(y_test, y_pred))

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# labels = ["Negatif (0)", "Positif (1)"]
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
# plt.title("Confusion Matrix - Logistic Regression")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.tight_layout()
# plt.show()

# # ====================
# # 5. Prediksi Semua Data
# # ====================
# df["PredictedLabel"] = model.predict(X)

# # ====================
# # 6. Analisis Aspek POSITIF SAJA
# # ====================
# def analisis_aspek_positif(df, review_col='CleanReview', label_col='PredictedLabel'):
#     # aspek_keywords = {
#     #     "pengiriman": ["pengiriman", "kirim", "datang","pengantar", "kurir", "antar", "sampai", "cepat", "lama", "lambat", "resi", "sampe"],
#     #     "pelayanan": ["pelayanan", "", "gercep", "layanan", "cs", "customer service", "respon", "tanggapan", "ramah", "komplain", "balas", "sopan", "cepat"],
#     #     "produk": ["produk", "cocok" "barang", "kualitas", "baik", "bagus", "asli", "ori", "original", "wangi", "recommend"],
#     #     "harga": ["harga", "murah", "diskon", "promo", "potongan", "worth it", "sesuai", "value", "terjangkau"],
#     #     "packing": ["packing", "paket", "bungkus", "kemasan", "wrap", "bubble", "rapi", "aman", "dus", "kotak", "plastik"]
#     # }
#     aspek_keywords = {
#     "pengiriman": [
#         "pengiriman", "kirim", "dikirim", "sampai", "datang", "pengantar", "kurir", "antar", 
#         "cepat", "tepat waktu", "on time", "kilat", "ekspres", "langsung sampai", 
#         "sampainya cepat", "proses cepat", "pengiriman bagus", "pengiriman mantap",
#         "nggak pake lama", "gak lama", "gak nunggu lama", "pengiriman lancar"
#     ],
#     "pelayanan": [
#         "pelayanan", "Terima Kasih", "Trima Kasih", "Terimakasih",
#         "Makasih", "Thankyou", "Thank You", "terima kasih", "Terima kasih",
#         "layanan", "gercep", "cs", "customer service", "respon cepat", 
#         "respon baik", "fast response", "tanggap", "ramah", "sopan", "baik", 
#         "melayani dengan baik", "bales cepat", "komunikatif", "dilayani dengan ramah", 
#         "responsif", "balesnya cepet", "admin baik", "admin ramah", "penjual baik", 
#         "penjual ramah", "penjual fast respon", "seller komunikatif", "penjual responsif"
#     ],
#     "produk": [
#         "produk", "barang", "kualitas bagus", "kualitas oke", "bagus", "asli", "ori", 
#         "original", "wangi", "bersih", "tidak cacat", "awet", "tahan lama", "kuat", 
#         "baik", "sesuai deskripsi", "mirip foto", "real pict", "real picture", "tidak mengecewakan", 
#         "top", "mantap", "mantul", "bagus banget", "bagus bgt", "rekomendasi", "recommended", 
#         "puas", "memuaskan", "worth it", "value for money", "barang oke", "barang keren",
#         "empuk", "nyaman", "enak", "utuh"
#     ],
#     "harga": [
#         "harga", "murah", "diskon", "promo", "potongan", "harga oke", "harga pas", 
#         "worth it", "sesuai harga", "value", "value for money", "terjangkau", "hemat", 
#         "ekonomis", "harga bersaing", "best deal", "murmer", "murah meriah", 
#         "harga mantap", "harganya oke", "harga murah", "good price", "harga bersahabat"
#     ],
#     "packing": [
#         "packing", "paket", "bungkus", "kemasan", "wrap", "bubble wrap", "bubble", 
#         "rapi", "aman", "packing aman", "packing rapi", "pengemasan bagus", 
#         "pengemasan rapi", "bungkus rapi", "kemasan aman", "kotak rapi", "plastik aman", 
#         "dus bagus", "dibungkus rapi", "packing mantap", "packing oke", "packing double", 
#         "anti pecah", "aman banget", "packingnya bagus", "dilapisi bubble wrap"
#     ]
# }


#     aspek_counter = defaultdict(int)
#     for _, row in df.iterrows():
#         if row[label_col] == 1:  # hanya ulasan positif
#             review = row[review_col]
#             for aspek, keywords in aspek_keywords.items():
#                 if any(kw in review for kw in keywords):
#                     aspek_counter[aspek] += 1
#     return aspek_counter

# # ====================
# # 7. Evaluasi Toko Berdasarkan Aspek Positif
# # ====================
# total_pos = (df["PredictedLabel"] == 1).sum()
# total_neg = (df["PredictedLabel"] == 0).sum()
# total_all = total_pos + total_neg
# persen_pos = round((total_pos / total_all) * 100, 2)

# toko_label = "Direkomendasikan" if persen_pos >= 60 else "Tidak Direkomendasikan"

# aspek_result = analisis_aspek_positif(df)
# aspek_tertinggi = max(aspek_result, key=aspek_result.get)
# jumlah_tertinggi = aspek_result[aspek_tertinggi]
# persen_tertinggi = round((jumlah_tertinggi / total_all) * 100, 2)

# # ====================
# # 8. Ringkasan
# # ====================
# print("\n=== PENILAIAN TOKO ===")
# print(f"- Total Ulasan: {total_all}")
# print(f"- Positif: {total_pos} ({persen_pos}%)")
# print(f"- Negatif: {total_neg} ({round((total_neg / total_all) * 100, 2)}%)")
# print(f"- Label Toko: {toko_label}")
# print(f"- Aspek yang menonjol (positif): '{aspek_tertinggi}' sebanyak {jumlah_tertinggi} review ({persen_tertinggi}%).")

# # ====================
# # 9. Visualisasi Aspek Positif
# # ====================
# labels = list(aspek_result.keys())
# counts = [aspek_result[k] for k in labels]

# plt.figure(figsize=(10, 6))
# plt.bar(labels, counts, color='green')
# plt.ylabel("Jumlah Review Positif")
# plt.title("Aspek Positif yang Paling Sering Disebutkan")
# plt.xticks(rotation=15)
# plt.tight_layout()
# plt.show()

# # ====================
# # 10. Lihat Kata-Kata TF-IDF Tertinggi
# # ====================
# import numpy as np

# # Jumlah rata-rata skor tiap kata
# tfidf_mean = np.asarray(X.mean(axis=0)).ravel()
# tfidf_vocab = vectorizer.get_feature_names_out()

# # Urutkan dari tertinggi
# top_n = 10
# top_indices = tfidf_mean.argsort()[::-1][:top_n]
# top_words = [(tfidf_vocab[i], tfidf_mean[i]) for i in top_indices]

# print("\n=== 10 Kata dengan Skor TF-IDF Tertinggi ===")
# for word, score in top_words:
#     print(f"{word}: {score:.4f}")

import pandas as pd
import re
import os
import joblib
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

# Unduh stopwords NLTK
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
# 2. Load & Clean Dataset
# ====================
df = pd.read_csv("scraping-result/Dataset.csv")
df.dropna(subset=["Review", "Rating"], inplace=True)
df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)
df["CleanReview"] = df["Review"].apply(clean_text)

# ====================
# 3. Vectorization & Model Training
# ====================
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df["CleanReview"])
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# ====================
# 4. Save Model & Vectorizer
# ====================
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/logistic_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("\n✅ Model dan vectorizer berhasil disimpan di folder 'model/'")

# ====================
# 5. Evaluation + Confusion Matrix
# ====================
y_pred = model.predict(X_test)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
labels = ["Negatif (0)", "Positif (1)"]
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ====================
# 6. Predict Semua Data
# ====================
df["PredictedLabel"] = model.predict(X)

# ====================
# 7. Analisis Aspek Positif
# ====================
def analisis_aspek_positif(df, review_col='CleanReview', label_col='PredictedLabel'):
    aspek_keywords = {
        "pengiriman": [
            "pengiriman", "kirim", "dikirim", "sampai", "datang", "pengantar", "kurir", "antar", 
            "cepat", "tepat waktu", "on time", "kilat", "ekspres", "langsung sampai", 
            "sampainya cepat", "proses cepat", "pengiriman bagus", "pengiriman mantap",
            "nggak pake lama", "gak lama", "gak nunggu lama", "pengiriman lancar"
        ],
        "pelayanan": [
            "pelayanan", "Terima Kasih", "Trima Kasih", "Terimakasih", "Makasih", "Thankyou", 
            "Thank You", "layanan", "gercep", "cs", "customer service", "respon cepat", 
            "respon baik", "fast response", "tanggap", "ramah", "sopan", "baik", 
            "melayani dengan baik", "bales cepat", "komunikatif", "dilayani dengan ramah", 
            "responsif", "balesnya cepet", "admin baik", "admin ramah", "penjual baik", 
            "penjual ramah", "penjual fast respon", "seller komunikatif", "penjual responsif"
        ],
        "produk": [
            "produk", "barang", "kualitas bagus", "kualitas oke", "bagus", "asli", "ori", 
            "original", "wangi", "bersih", "tidak cacat", "awet", "tahan lama", "kuat", 
            "baik", "sesuai deskripsi", "mirip foto", "real pict", "real picture", 
            "tidak mengecewakan", "top", "mantap", "mantul", "bagus banget", "bagus bgt", 
            "rekomendasi", "recommended", "puas", "memuaskan", "worth it", "value for money", 
            "barang oke", "barang keren", "empuk", "nyaman", "enak", "utuh"
        ],
        "harga": [
            "harga", "murah", "diskon", "promo", "potongan", "harga oke", "harga pas", 
            "worth it", "sesuai harga", "value", "value for money", "terjangkau", "hemat", 
            "ekonomis", "harga bersaing", "best deal", "murmer", "murah meriah", 
            "harga mantap", "harganya oke", "harga murah", "good price", "harga bersahabat"
        ],
        "packing": [
            "packing", "paket", "bungkus", "kemasan", "wrap", "bubble wrap", "bubble", 
            "rapi", "aman", "packing aman", "packing rapi", "pengemasan bagus", 
            "pengemasan rapi", "bungkus rapi", "kemasan aman", "kotak rapi", "plastik aman", 
            "dus bagus", "dibungkus rapi", "packing mantap", "packing oke", "packing double", 
            "anti pecah", "aman banget", "packingnya bagus", "dilapisi bubble wrap"
        ]
    }

    aspek_counter = defaultdict(int)
    for _, row in df.iterrows():
        if row[label_col] == 1:  # hanya ulasan positif
            review = row[review_col]
            for aspek, keywords in aspek_keywords.items():
                if any(kw in review for kw in keywords):
                    aspek_counter[aspek] += 1
    return aspek_counter

aspek_result = analisis_aspek_positif(df)
total_pos = (df["PredictedLabel"] == 1).sum()
total_neg = (df["PredictedLabel"] == 0).sum()
total_all = total_pos + total_neg
persen_pos = round((total_pos / total_all) * 100, 2)
toko_label = "Direkomendasikan" if persen_pos >= 60 else "Tidak Direkomendasikan"
aspek_tertinggi = max(aspek_result, key=aspek_result.get)
jumlah_tertinggi = aspek_result[aspek_tertinggi]
persen_tertinggi = round((jumlah_tertinggi / total_all) * 100, 2)

# ====================
# 8. Ringkasan
# ====================
print("\n=== PENILAIAN TOKO ===")
print(f"- Total Ulasan: {total_all}")
print(f"- Positif: {total_pos} ({persen_pos}%)")
print(f"- Negatif: {total_neg} ({round((total_neg / total_all) * 100, 2)}%)")
print(f"- Label Toko: {toko_label}")
print(f"- Aspek yang menonjol (positif): '{aspek_tertinggi}' sebanyak {jumlah_tertinggi} review ({persen_tertinggi}%).")

# ====================
# 9. Visualisasi Aspek Positif
# ====================
labels = list(aspek_result.keys())
counts = [aspek_result[k] for k in labels]

plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color='green')
plt.ylabel("Jumlah Review Positif")
plt.title("Aspek Positif yang Paling Sering Disebutkan")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ====================
# 10. Lihat Kata-Kata TF-IDF Tertinggi
# ====================
tfidf_mean = np.asarray(X.mean(axis=0)).ravel()
tfidf_vocab = vectorizer.get_feature_names_out()
top_n = 10
top_indices = tfidf_mean.argsort()[::-1][:top_n]
top_words = [(tfidf_vocab[i], tfidf_mean[i]) for i in top_indices]

print("\n=== 10 Kata dengan Skor TF-IDF Tertinggi ===")
for word, score in top_words:
    print(f"{word}: {score:.4f}")
