# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.svm import SVC
# # from sklearn.metrics import classification_report
# # from utils.preprocess import clean_text, vectorize_text

# # def train_svm():
# #     # Baca data
# #     df = pd.read_csv("data/ulasan_produk.csv")
# #     df.dropna(subset=["Komentar", "Rating"], inplace=True)

# #     # Labeling: rating >= 4 = positif (1), <4 = negatif (0)
# #     df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)

# #     # Preprocessing teks
# #     df["CleanKomentar"] = df["Komentar"].apply(clean_text)
# #     X, vectorizer = vectorize_text(df["CleanKomentar"])
# #     y = df["Label"]

# #     # Split train/test
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #     # Model SVM
# #     model = SVC(kernel='linear')
# #     model.fit(X_train, y_train)

# #     # Evaluasi
# #     y_pred = model.predict(X_test)
# #     print("📊 Hasil Evaluasi Model SVM:")
# #     print(classification_report(y_test, y_pred))

# #     return model, vectorizer
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report
# from utils.preprocess import clean_text, vectorize_text

# def train_svm():
#     # Baca data
#     df = pd.read_csv("data/ulasan_produk.csv")
#     df.dropna(subset=["ulasan", "rating"], inplace=True)

#     # Labeling: rating >= 4 = positif (1), <4 = negatif (0)
#     df["Label"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

#     # Preprocessing teks
#     df["CleanUlasan"] = df["ulasan"].apply(clean_text)
#     X, vectorizer = vectorize_text(df["CleanUlasan"])
#     y = df["Label"]

#     # Split train/test
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Model SVM
#     model = SVC(kernel='linear')
#     model.fit(X_train, y_train)

#     # Evaluasi
#     y_pred = model.predict(X_test)
#     print("📊 Hasil Evaluasi Model SVM:")
#     print(classification_report(y_test, y_pred))

#     return model, vectorizer

# import os
# import sys

# # Tambahkan path supaya bisa import dari utils/
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report
# from utils.preprocess import clean_text, vectorize_text

# def train_svm():
#     # Baca data
#     df = pd.read_csv("dataset/shopee100k.csv")
#     df.dropna(subset=["ulasan", "rating"], inplace=True)

#     # Labeling: rating >= 4 = positif (1), <4 = negatif (0)
#     df["Label"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

#     # Preprocessing teks
#     df["CleanUlasan"] = df["ulasan"].apply(clean_text)
#     X, vectorizer = vectorize_text(df["CleanUlasan"])
#     y = df["Label"]

#     # Split train/test
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Model SVM
#     model = SVC(kernel='linear')
#     model.fit(X_train, y_train)

#     # Evaluasi
#     y_pred = model.predict(X_test)
#     print("📊 Hasil Evaluasi Model SVM:")
#     print(classification_report(y_test, y_pred))

#     return model, vectorizer

# if __name__ == "__main__":
#     train_svm()
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# --- Download stopwords Indonesia (cukup sekali) ---
nltk.download('stopwords')
stop_words_ind = stopwords.words('indonesian')

# --- Preprocessing function ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)      # Hapus simbol dan angka
    text = re.sub(r'\s+', ' ', text).strip()     # Hapus spasi ganda
    return text

# --- Load data ---
df = pd.read_csv("scraping-result/shopee_ratingnew.csv")
df.dropna(subset=["Review", "Rating"], inplace=True)

# --- Labeling: Rating >= 4 = Positif (1), < 4 = Negatif (0) ---
df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)

# --- Preprocessing text ---
df["CleanReview"] = df["Review"].apply(clean_text)

# --- TF-IDF Vectorization ---
vectorizer = TfidfVectorizer(stop_words=stop_words_ind)
X = vectorizer.fit_transform(df["CleanReview"])
y = df["Label"]

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- SVM model ---
model = SVC(kernel='linear', class_weight='balanced')  # linear kernel cocok untuk teks
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
print("📊 Hasil Evaluasi Model SVM:")
print(classification_report(y_test, y_pred, zero_division=0))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
labels = ['Negatif (0)', 'Positif (1)']

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix SVM')
plt.tight_layout()
plt.show()
