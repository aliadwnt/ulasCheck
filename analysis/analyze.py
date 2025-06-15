# import pandas as pd
# import joblib
# from collections import defaultdict

# # Load pipeline (vectorizer + model)
# pipeline = joblib.load("model/pipeline.pkl")

# def analyze_data(csv_path):
#     df = pd.read_csv(csv_path)

#     # Bersihkan teks
#     def clean_text(text):
#         if not isinstance(text, str):
#             return ""
#         text = text.lower()
#         text = re.sub(r'[^a-zA-Z\s]', '', text)
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text

#     df["CleanReview"] = df["Review"].apply(clean_text)
    
#     # Prediksi label
#     df["PredictedLabel"] = pipeline.predict(df["CleanReview"])

#     # Hitung total positif/negatif
#     total_pos = (df["PredictedLabel"] == 1).sum()
#     total_neg = (df["PredictedLabel"] == 0).sum()
#     total_all = total_pos + total_neg
#     persen_pos = round((total_pos / total_all) * 100, 2)

#     label_toko = "Direkomendasikan" if persen_pos >= 60 else "Tidak Direkomendasikan"

#     # Analisis aspek
#     aspek_keywords = {
#         "produk": ["bagus", "mantap", "kualitas", "oke", "puas", "ori"],
#         "harga": ["murah", "terjangkau", "diskon", "worth", "harga"],
#         "pengiriman": ["cepat", "sampai", "pengiriman", "kurir"],
#         "pelayanan": ["ramah", "respon", "fast", "admin"],
#         "packing": ["rapi", "aman", "packing", "bubble"]
#     }

#     aspek_counter = defaultdict(int)
#     for _, row in df.iterrows():
#         if row["PredictedLabel"] == 1:
#             for aspek, keywords in aspek_keywords.items():
#                 if any(kw in row["CleanReview"] for kw in keywords):
#                     aspek_counter[aspek] += 1

#     aspek_tertinggi = max(aspek_counter, key=aspek_counter.get, default="-")
#     jumlah_tertinggi = aspek_counter.get(aspek_tertinggi, 0)
#     persen_tertinggi = round((jumlah_tertinggi / total_all) * 100, 2) if total_all > 0 else 0

#     return {
#         "total": total_all,
#         "positif": total_pos,
#         "negatif": total_neg,
#         "persen_positif": persen_pos,
#         "label_toko": label_toko,
#         "aspek_tertinggi": aspek_tertinggi,
#         "jumlah_aspek": jumlah_tertinggi,
#         "persen_aspek": persen_tertinggi
#     }
# analysis/analyze.py
# import pandas as pd
# import joblib
# from analysis.aspek_analysis import analisis_aspek_positif

# pipeline = joblib.load("model/pipeline.pkl")

# def analyze_data(csv_path):
#     df = pd.read_csv(csv_path)

#     # Preprocess
#     df['CleanReview'] = df['Review'].fillna('').str.lower()

#     # Predict
#     df['PredictedLabel'] = pipeline.predict(df['CleanReview'])

#     # Statistik prediksi
#     total_pos = (df["PredictedLabel"] == 1).sum()
#     total_neg = (df["PredictedLabel"] == 0).sum()
#     total_all = len(df)
#     persen_pos = round((total_pos / total_all) * 100, 2)
#     toko_label = "Direkomendasikan" if persen_pos >= 60 else "Tidak Direkomendasikan"

#     # Analisis aspek
#     aspek_result = analisis_aspek_positif(df)
#     aspek_tertinggi = max(aspek_result, key=aspek_result.get)
#     jumlah_tertinggi = aspek_result[aspek_tertinggi]
#     persen_tertinggi = round((jumlah_tertinggi / total_all) * 100, 2)

#     return {
#         "total": total_all,
#         "positif": total_pos,
#         "negatif": total_neg,
#         "persen_positif": persen_pos,
#         "label": toko_label,
#         "aspek_tertinggi": aspek_tertinggi,
#         "jumlah_aspek": jumlah_tertinggi,
#         "persen_aspek": persen_tertinggi,
#     }
import pandas as pd
import joblib
from analysis.analyze import analisis_aspek_positif

# Load model pipeline
pipeline = joblib.load("model/pipeline.pkl")

def analyze_data(csv_path):
    try:
        # Baca file CSV
        df = pd.read_csv(csv_path)

        # Preprocessing: ubah ke huruf kecil dan isi NaN dengan string kosong
        df['CleanReview'] = df['Review'].fillna('').str.lower()

        # Prediksi sentimen: 1 = Positif, 0 = Negatif
        df['PredictedLabel'] = pipeline.predict(df['CleanReview'])

        # Hitung statistik
        total_review = len(df)
        total_positif = (df['PredictedLabel'] == 1).sum()
        total_negatif = (df['PredictedLabel'] == 0).sum()
        persen_positif = round((total_positif / total_review) * 100, 2)
        label_toko = "Toko Direkomendasikan" if persen_positif >= 60 else "Toko Tidak Direkomendasikan"

        # Analisis aspek positif
        aspek_result = analisis_aspek_positif(df)

        # Ambil aspek dengan jumlah tertinggi
        if aspek_result:
            aspek_tertinggi = max(aspek_result, key=aspek_result.get)
            jumlah_tertinggi = aspek_result[aspek_tertinggi]
            persen_tertinggi = round((jumlah_tertinggi / total_review) * 100, 2)
        else:
            aspek_tertinggi = "-"
            jumlah_tertinggi = 0
            persen_tertinggi = 0.0

        return {
            "total": total_review,
            "positif": total_positif,
            "negatif": total_negatif,
            "persen_pos": persen_positif,
            "label_toko": label_toko,
            "aspek": aspek_result,
            "aspek_tertinggi": aspek_tertinggi,
            "jumlah_aspek": jumlah_tertinggi,
            "persen_aspek": persen_tertinggi,
        }

    except Exception as e:
        print(f"[ERROR] Gagal menganalisis data: {e}")
        return None
