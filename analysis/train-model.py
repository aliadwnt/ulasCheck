import pandas as pd
import re
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Download stopwords jika belum
nltk.download('stopwords')
stop_words = stopwords.words('indonesian')

# --- Preprocessing ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Load Dataset ---
df = pd.read_csv("scraping-result/dataset.csv")
df.dropna(subset=["Review", "Rating"], inplace=True)
df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)
df["CleanReview"] = df["Review"].apply(clean_text)

# --- Build Pipeline ---
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])

pipeline.fit(df["CleanReview"], df["Label"])

# --- Save pipeline as .pkl ---
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/pipeline.pkl")
print("✅ Pipeline berhasil disimpan sebagai model/pipeline.pkl")
