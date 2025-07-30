from flask import render_template, request, redirect, url_for, flash, session, send_file
from app.models.reviewModel import Review
from app.models.userModel import User
from app.utils.scraper import shopee
from app.analyze import analyze_data, get_keyword_frequency
from app.extensions import db
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import io, csv, os, re, unicodedata
import pandas as pd
from app.models.predictionModel import Prediction

def inject_user_and_now():
    user = None
    if "user_id" in session:
        user = User.query.get(session["user_id"])
    return dict(current_user=user, now=datetime.now())

def index():
    if request.method == "POST":
        link = request.form.get("link")
        cookie_path = "cookies/cookie.json"
        review_id, message = shopee(link, cookie_path)

        if review_id:
            return redirect(f"/review/{review_id}")
        else:
            flash(message or "Gagal mengambil data review.", "danger")
            return redirect("/")
            
    return render_template("pages/public/dashboard.html")

def scrapeToko():
    if request.method == "POST":
        link = request.form.get("link")
        cookie_path = "cookies/cookie.json"
        review_id, message = shopee(link, cookie_path)

        if review_id:
            return redirect(f"/review/{review_id}")
        else:
            flash(message or "Gagal mengambil data review toko.", "danger")
            return redirect("/")
            
    return render_template("pages/public/scrape-toko.html")

def show_review(id):
    review = Review.query.get_or_404(id)
    scraped_data = []
    keyword_freq = {}

    if review.file_data:
        try:
            file_stream = io.StringIO(review.file_data.decode("utf-8"))
            reader = csv.DictReader(file_stream)
            scraped_data = list(reader)

            # Ambil semua review
            review_texts = [row['Review'] for row in scraped_data if 'Review' in row]

            # Hitung frekuensi kata
            keyword_freq = get_keyword_frequency(review_texts)

        except Exception:
            flash("Gagal membaca data ulasan", "danger")

    return render_template("pages/public/dashboard.html", scraped_data=scraped_data, review=review, keyword_freq=keyword_freq)
    
def download_file(id):
    review = Review.query.get_or_404(id)
    return send_file(
        io.BytesIO(review.file_data),
        mimetype='text/csv',
        as_attachment=True,
        download_name=review.file
    )

def analyze_file(id):
    review = Review.query.get_or_404(id)

    if not review.file_data:
        flash("Data ulasan tidak ditemukan.", "danger")
        return redirect(f"/review/{id}")

    try:
        file_stream = io.StringIO(review.file_data.decode("utf-8"))
        temp_df = pd.read_csv(file_stream)
    except Exception as e:
        flash(f"Gagal membaca data: {str(e)}", "danger")
        return redirect(f"/review/{id}")

    try:
        hasil = analyze_data(temp_df)

        # Simpan hasil analisis ke database
        new_prediction = Prediction(
            review_id=review.id,
            total_all=hasil.get("total", 0),
            total_pos=hasil.get("positif", 0),
            total_neg=hasil.get("negatif", 0),
            persen_pos=hasil.get("persen_pos", 0),
            toko_label=hasil.get("label_toko", "Tidak Diketahui"),
            aspek=hasil.get("aspek_tertinggi", "-"),
            persen_tertinggi=hasil.get("persen_aspek", 0)
        )
        db.session.add(new_prediction)
        db.session.commit()

        # Cari ulasan positif terkait aspek tertinggi
        aspek_tertinggi = hasil.get("aspek_tertinggi", "").lower()

        aspek_keywords = {
            "produk": ["produk", "barang", "kualitas", "ukuran", "warna"],
            "pengiriman": ["pengiriman", "cepat", "packing", "kurir", "sampai"],
            "pelayanan": ["pelayanan", "respon", "penjual", "ramah", "tanggap"],
            "harga": ["harga", "murah", "diskon", "promo", "terjangkau"],
            "packing": ["packing", "kemasan", "bungkus", "rapi", "aman"]
        }

        keywords = aspek_keywords.get(aspek_tertinggi, [])
        highlighted_reviews = []

        # Bersihkan review dulu jika belum ada kolom CleanReview
        def clean_review(text):
            # Pastikan teks string
            text = str(text).lower()

            # Normalisasi karakter unicode (hilangkan simbol aneh & emot)
            text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

            # Ganti karakter non-huruf (angka, simbol) dengan spasi
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)

            # Hapus spasi berlebih
            text = re.sub(r'\s+', ' ', text).strip()

            return text

        if "CleanReview" not in temp_df.columns:
            temp_df["CleanReview"] = temp_df["Review"].apply(clean_review)

        for _, row in temp_df.iterrows():
            review_text = str(row.get("CleanReview", "")).strip()
            if row.get("Rating", 0) >= 4:
                if (
                    any(kw in review_text for kw in keywords)
                    and len(review_text.split()) > 15
                ):
                    highlighted_reviews.append({
                        "Username": row.get("Username", "-"),
                        "Review": row.get("Review", "-"),  # Tampilkan versi asli
                        "Rating": row.get("Rating", 0),
                        "Cleaned": review_text  # Ditampilkan kalau mau
                    })

        highlighted_reviews = highlighted_reviews[:5]


    except Exception as e:
        db.session.rollback()
        flash(f"Gagal melakukan analisis: {str(e)}", "danger")
        return redirect(f"/review/{id}")

    return render_template(
        "pages/public/dashboard.html",
        review=review,
        scraped_data=temp_df.to_dict(orient="records"),
        result={
            "total_all": new_prediction.total_all,
            "total_pos": new_prediction.total_pos,
            "total_neg": new_prediction.total_neg,
            "persen_pos": new_prediction.persen_pos,
            "toko_label": new_prediction.toko_label,
            "aspek": new_prediction.aspek,
            "persen_tertinggi": new_prediction.persen_tertinggi,
            "aspek_persen": hasil.get("aspek_persen", {})
        },
        highlighted_reviews=highlighted_reviews,
        aspek_tertinggi=aspek_tertinggi
    )

def cancel(id):
    flash("❌ Analisis dibatalkan", "info")
    return redirect("/")

def about_us():
    return render_template("pages/public/about-us.html")

def scrapeProduk():
    return render_template("pages/public/scraping-produk.html")