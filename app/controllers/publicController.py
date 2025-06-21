from flask import render_template, request, redirect, url_for, flash, session, send_file
from app.models.reviewModel import Review
from app.models.userModel import User
from app.utils.scraper import shopee
from app.analyze import analyze_data
from app.extensions import db
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import io, csv, os
import pandas as pd

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

def show_review(id):
    review = Review.query.get_or_404(id)
    scraped_data = []

    if review.file_data:
        try:
            file_stream = io.StringIO(review.file_data.decode("utf-8"))
            reader = csv.DictReader(file_stream)
            scraped_data = list(reader)
        except Exception:
            flash("Gagal membaca data ulasan", "danger")

    return render_template("pages/public/dashboard.html", scraped_data=scraped_data, review=review)

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

    # Pastikan review memiliki file_data
    if not review.file_data:
        flash("Data ulasan tidak ditemukan.", "danger")
        return redirect(f"/review/{id}")

    # Membaca data CSV dari file_data yang ada dalam review
    file_stream = io.StringIO(review.file_data.decode("utf-8"))
    temp_df = pd.read_csv(file_stream)

    # Panggil fungsi analisis untuk mendapatkan hasil analisis langsung menggunakan DataFrame
    hasil = analyze_data(temp_df)

    return render_template("pages/public/dashboard.html", result={
        "total_all": hasil["total"],
        "total_pos": hasil["positif"],
        "total_neg": hasil["negatif"],
        "persen_pos": hasil["persen_pos"],
        "toko_label": hasil["label_toko"],
        "aspek": hasil["aspek_tertinggi"],
        "persen_tertinggi": hasil["persen_aspek"],
        "scraped_data": temp_df.to_dict(orient='records')  # Menambahkan data review yang bersih ke template
    })

def cancel(id):
    flash("❌ Analisis dibatalkan", "info")
    return redirect("/")

def about_us():
    return render_template("pages/public/about-us.html")
