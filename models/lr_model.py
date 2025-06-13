# import pandas as pd
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import classification_report
# import nltk
# from nltk.corpus import stopwords

# # --- Download stopwords Indo (hanya perlu sekali) ---
# nltk.download('stopwords')
# stop_words_ind = stopwords.words('indonesian')

# # --- Preprocessing ---
# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hilangkan simbol & angka
#     text = re.sub(r'\s+', ' ', text).strip()  # Hilangkan spasi ganda
#     return text

# # --- Load data ---
# df = pd.read_csv("scraping-result/shopee_ratingnew.csv")  # Pastikan path benar
# df.dropna(subset=["Review", "Rating"], inplace=True)

# # --- Labeling: Rating >= 4 = Positif (1), < 4 = Negatif (0) ---
# df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)

# # --- Preprocess teks ---
# df["CleanReview"] = df["Review"].apply(clean_text)

# # --- TF-IDF Vectorization with Indonesian stopwords ---
# vectorizer = TfidfVectorizer(stop_words=stop_words_ind)
# X = vectorizer.fit_transform(df["CleanReview"])
# y = df["Label"]

# # --- Split data ---
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # --- Model Logistic Regression ---
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # --- Evaluasi Model ---
# y_pred = model.predict(X_test)
# print("📊 Hasil Evaluasi Model Logistic Regression:")
# print(classification_report(y_test, y_pred))
# import pandas as pd
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import classification_report
# import nltk
# from nltk.corpus import stopwords

# # --- Download stopwords Indonesia (cukup sekali) ---
# nltk.download('stopwords')
# stop_words_ind = stopwords.words('indonesian')

# # --- Preprocessing function ---
# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)      # Hilangkan simbol & angka
#     text = re.sub(r'\s+', ' ', text).strip()     # Hilangkan spasi ganda
#     return text

# # --- Load data ---
# df = pd.read_csv("scraping-result/shopee_ratingnew.csv")
# df.dropna(subset=["Review", "Rating"], inplace=True)

# # --- Labeling: Rating >= 4 = Positif (1), < 4 = Negatif (0) ---
# df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)

# # --- Preprocess teks ---
# df["CleanReview"] = df["Review"].apply(clean_text)

# # --- TF-IDF Vectorization ---
# vectorizer = TfidfVectorizer(stop_words=stop_words_ind)
# X = vectorizer.fit_transform(df["CleanReview"])
# y = df["Label"]

# # --- Train-test split ---
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # --- Train Logistic Regression with balanced class weight ---
# model = LogisticRegression(class_weight='balanced', max_iter=1000)
# model.fit(X_train, y_train)

# # --- Evaluation ---
# y_pred = model.predict(X_test)
# print("📊 Hasil Evaluasi Model Logistic Regression:")
# print(classification_report(y_test, y_pred, zero_division=0))

# import pandas as pd
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import classification_report, confusion_matrix
# import nltk
# from nltk.corpus import stopwords
# import matplotlib.pyplot as plt
# import seaborn as sns

# # --- Download stopwords Indonesia (cukup sekali) ---
# nltk.download('stopwords')
# stop_words_ind = stopwords.words('indonesian')

# # --- Preprocessing function ---
# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)      # Hapus simbol dan angka
#     text = re.sub(r'\s+', ' ', text).strip()     # Hapus spasi ganda
#     return text

# # --- Load data ---
# df = pd.read_csv("scraping-result/shopee_ratingnew.csv")
# df.dropna(subset=["Review", "Rating"], inplace=True)

# # --- Labeling: Rating >= 4 = Positif (1), < 4 = Negatif (0) ---
# df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)

# # --- Preprocessing text ---
# df["CleanReview"] = df["Review"].apply(clean_text)

# # --- TF-IDF Vectorization ---
# vectorizer = TfidfVectorizer(stop_words=stop_words_ind)
# X = vectorizer.fit_transform(df["CleanReview"])
# y = df["Label"]

# # --- Split data ---
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # --- Logistic Regression model ---
# model = LogisticRegression(class_weight='balanced', max_iter=1000)
# model.fit(X_train, y_train)

# # --- Evaluation ---
# y_pred = model.predict(X_test)
# print("📊 Hasil Evaluasi Model Logistic Regression:")
# print(classification_report(y_test, y_pred, zero_division=0))

# # --- Confusion Matrix ---
# cm = confusion_matrix(y_test, y_pred)
# labels = ['Negatif (0)', 'Positif (1)']

# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix Logistic Regression')
# plt.tight_layout()
# plt.show()
# model/lr_model.py
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')
stop_words = stopwords.words('indonesian')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df = pd.read_csv("scraping-result/shopee_ratingnew.csv")
df.dropna(subset=["Review", "Rating"], inplace=True)
df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)
df["CleanReview"] = df["Review"].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df["CleanReview"])
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix Logistic Regression")
plt.show()
