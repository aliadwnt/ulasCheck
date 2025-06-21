from flask import Blueprint, render_template, request, redirect, url_for, flash, session, send_file
from app.models.reviewModel import Review
from app.models.userModel import User
from app.utils.scraper import shopee
from app.analyze import analyze_data
from app.extensions import db
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import io, csv, os
import pandas as pd

public = Blueprint("public", __name__)  # ✅ Diubah dari 'main' ke 'public'

@public.context_processor
def inject_user_and_now():
    user = None
    if "user_id" in session:
        user = User.query.get(session["user_id"])
    return dict(current_user=user, now=datetime.now())

@public.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        link = request.form.get("link")
        cookie_path = "cookies/cookie.json"

        scraped_data, message = shopee(link, cookie_path)

        if scraped_data:
            try:
                review = Review(
                    shop_id=parse_qs(urlparse(link).query).get("shop_id", ["unknown"])[0],
                    file="rating.csv",
                    file_data=None
                )
                db.session.add(review)
                db.session.commit()
                return redirect(f"/review/{review.id}")
            except Exception as e:
                db.session.rollback()
                flash(f"Gagal menyimpan ke database: {e}", "danger")
                return redirect("/")
        else:
            flash(message or "Gagal mengambil data review.", "danger")
            return redirect("/")

    return render_template("pages/public/dashboard.html")

@public.route("/review/<int:id>")
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

@public.route("/download/<int:id>", methods=["POST"])
def download_file(id):
    review = Review.query.get_or_404(id)
    return send_file(
        io.BytesIO(review.file_data),
        mimetype='text/csv',
        as_attachment=True,
        download_name=review.file
    )

@public.route("/analyze/<int:id>", methods=["POST"])
def analyze_file(id):
    review = Review.query.get_or_404(id)
    file_stream = io.StringIO(review.file_data.decode("utf-8"))
    temp_df = pd.read_csv(file_stream)
    temp_csv_path = "temp_analysis.csv"
    temp_df.to_csv(temp_csv_path, index=False)
    hasil = analyze_data(temp_csv_path)
    os.remove(temp_csv_path)

    return render_template("pages/public/dashboard.html", result={
        "total_all": hasil["total"],
        "total_pos": hasil["positif"],
        "total_neg": hasil["negatif"],
        "persen_pos": hasil["persen_pos"],
        "toko_label": hasil["label_toko"],
        "aspek": hasil["aspek_tertinggi"],
        "persen_tertinggi": hasil["persen_aspek"]
    })

@public.route("/cancel/<int:id>")
def cancel(id):
    flash("❌ Analisis dibatalkan", "info")
    return redirect("/")

@public.route("/about-us")
def aboutUs():
    return render_template("pages/public/about-us.html")
