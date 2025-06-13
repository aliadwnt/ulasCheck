import pandas as pd
import random
from datetime import datetime, timedelta

usernames = [f"user_{i}" for i in range(1, 25001)]
produk_list = [
    "Sepatu Sneakers", "Tas Wanita", "Smartphone Android", "Kemeja Pria", "Jam Tangan Digital",
    "Mouse Wireless", "Headset Gaming", "Buku Catatan", "Powerbank", "Makeup Set",
    "Skincare Serum", "Kaos Polos", "Meja Belajar", "Kursi Lipat", "Lampu Tidur",
    "Rak Sepatu", "Parfum Wanita", "Celana Jeans", "Keyboard Mechanical", "Flashdisk 64GB"
]

ulasan_positif = [
    "Produk sangat bagus dan sesuai deskripsi!", "Pengiriman cepat dan aman.", "Barang original dan berkualitas.",
    "Penjual sangat ramah dan fast respon.", "Harga terjangkau dengan kualitas oke.",
    "Packing rapi dan aman sampai tujuan.", "Sangat puas dengan pelayanan toko ini.", "Sudah beli berkali-kali di sini dan selalu puas.",
    "Kualitas di atas ekspektasi saya!", "Sangat direkomendasikan untuk semua orang.",
    "Barang datang lebih cepat dari perkiraan.", "Ukuran pas dan sesuai pesanan.", "Warna dan bentuk persis seperti di gambar.",
    "Top banget! Bakal order lagi nanti.", "Sesuai dengan rating dan review yang saya baca.",
    "Mantap, gak nyesel beli di sini.", "Puas banget, terima kasih seller!", "Suka banget sama produk ini.",
    "Worth it banget untuk harga segini.", "Pelayanan cepat, packing aman, produk oke!"
]

ulasan_negatif = [
    "Produk tidak sesuai gambar.", "Pengiriman sangat lama.", "Barang rusak saat diterima.",
    "Packing asal-asalan.", "Penjual tidak merespon chat.", "Barang tidak berfungsi dengan baik.",
    "Ukuran tidak sesuai deskripsi.", "Kualitas sangat buruk.", "Sangat kecewa dengan produk ini.",
    "Terlalu mahal untuk kualitas seperti ini.", "Tidak direkomendasikan sama sekali.",
    "Barang palsu, bukan original.", "Produk kotor dan tidak layak pakai.", "Warna tidak sesuai pesanan.",
    "Sudah komplain tapi tidak ada solusi.", "Servis buruk, tidak profesional.",
    "Barang cacat, sangat mengecewakan.", "Rating tinggi tapi kualitas rendah.",
    "Beli karena review bagus, ternyata zonk.", "Baru dipakai sebentar sudah rusak."
]

def generate_random_date(start_date, end_date):
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")

# Data positif
data_positif = [{
    "Username": random.choice(usernames),
    "Produk": random.choice(produk_list),
    "Review": random.choice(ulasan_positif),
    "Rating": random.randint(4, 5),
    "ReviewAt": generate_random_date(datetime(2023, 1, 1), datetime(2024, 12, 31))
} for _ in range(20000)]

# Data negatif
data_negatif = [{
    "Username": random.choice(usernames),
    "Produk": random.choice(produk_list),
    "Review": random.choice(ulasan_negatif),
    "Rating": random.randint(1, 3),
    "ReviewAt": generate_random_date(datetime(2023, 1, 1), datetime(2024, 12, 31))
} for _ in range(5000)]

# Gabungkan dan simpan
full_data = pd.DataFrame(data_positif + data_negatif)
full_data.to_csv("generated_shopee_reviews.csv", index=False)
