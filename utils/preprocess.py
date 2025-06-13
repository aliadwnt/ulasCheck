# import re
# from sklearn.feature_extraction.text import TfidfVectorizer

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
#     return text

# def vectorize_text(corpus):
#     vectorizer = TfidfVectorizer(max_features=1000)
#     X = vectorizer.fit_transform(corpus)
#     return X, vectorizerimport re
# from sklearn.feature_extraction.text import TfidfVectorizer

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hilangkan simbol dan angka
#     text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi berlebih
#     return text

# def vectorize_text(text_series):
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(text_series)
#     return X, vectorizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

def vectorize_text(text_series):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_series)
    return X, vectorizer