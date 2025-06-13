import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Buat folder dataset jika belum ada ---
os.makedirs("dataset", exist_ok=True)

# --- Load data ---
df = pd.read_csv("scraping-result/dataset.csv")

# --- Bersihkan data kosong ---
df.dropna(subset=["Review", "Rating"], inplace=True)

# --- Labeling: Rating >= 4 dianggap positif (1), sisanya negatif (0) ---
df["Label"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)

# --- Split data: 80% training, 20% testing ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# --- Simpan ke CSV ---
train_df.to_csv("dataset/train-dataset.csv", index=False)
test_df.to_csv("dataset/test-dataset.csv", index=False)

print("✅ Dataset berhasil dipisahkan dan disimpan:")
print(f"- Jumlah data total   : {len(df)}")
print(f"- Data training (80%) : {len(train_df)} disimpan di 'dataset/train-dataset.csv'")
print(f"- Data testing (20%)  : {len(test_df)} disimpan di 'dataset/test-dataset.csv'")
