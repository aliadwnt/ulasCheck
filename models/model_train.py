from utils.preprocess import clean_text, vectorize_text
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def train_svm():
    df = pd.read_csv("data/ulasan_produk.csv")
    df.dropna(subset=["ulasan", "rating"], inplace=True)
    df["Label"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)
    df["CleanUlasan"] = df["ulasan"].apply(clean_text)
    X, vectorizer = vectorize_text(df["CleanUlasan"])
    y = df["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("📊 Hasil Evaluasi Model SVM:")
    print(classification_report(y_test, y_pred))
    return model, vectorizer
