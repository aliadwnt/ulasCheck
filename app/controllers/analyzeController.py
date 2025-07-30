from flask import request, redirect, url_for, send_file, flash, render_template
import os
from app.services.analyze_service import run_analysis
import pandas as pd

def analyze_review(review_id):
    try:
        print(f"🔍 Memulai analisis untuk review ID {review_id}")

        file_path = f"data/scraped/review_{review_id}.csv"
        if not os.path.exists(file_path):
            flash("File review tidak ditemukan.", "danger")
            return redirect(url_for("public.show_review", id=review_id))

        df = pd.read_csv(file_path)

        hasil = run_analysis(df)

        result = {
            "total_all": hasil.get("total", 0),
            "total_pos": hasil.get("positif", 0),
            "total_neg": hasil.get("negatif", 0),
            "persen_pos": hasil.get("persen_pos", 0),
            "toko_label": hasil.get("label_toko", "Tidak Diketahui"),
            "aspek": hasil.get("aspek_tertinggi", "-"),
            "persen_tertinggi": hasil.get("persen_aspek", 0),
            "aspek_persen": hasil.get("aspek_persen", {}),
            "scraped_data": df.to_dict(orient="records")
        }

        return render_template("pages/public/dashboard.html", result=result, scraped_data=result["scraped_data"], review_id=review_id)

    except Exception as e:
        flash(f"Terjadi kesalahan saat analisis: {str(e)}", "danger")
        return redirect(url_for("public.show_review", id=review_id))


def download_review(review_id):
    file_path = f"data/scraped/review_{review_id}.csv"
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash("File tidak ditemukan.")
        return redirect(url_for("public.show_review", id=review_id))


def cancel_review(review_id):
    try:
        os.remove(f"data/scraped/review_{review_id}.csv")
        flash("Proses dibatalkan.")
    except FileNotFoundError:
        flash("File tidak ditemukan atau sudah dihapus.")
    return redirect(url_for("public.show_review", id=review_id))
