from flask import render_template, abort, flash, redirect, send_file, request, session, url_for, jsonify
from app.models.reviewModel import Review
import pandas as pd
from app import db
import io, pytz
from datetime import datetime

# ✅ Tampilkan semua riwayat review
def show_reviews():
    reviews = Review.query.order_by(Review.updatedAt.desc()).all()
    return render_template("pages/admin/history.html", reviews=reviews)

# ✅ Download file review sebagai CSV
def download_file(id):
    review = Review.query.get_or_404(id)
    if not review.file_data:
        flash("File tidak ditemukan.", "error")
        return redirect("/admin/history")

    return send_file(
        io.BytesIO(review.file_data),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"review_{review.id}.csv"
    )

# ✅ Halaman dataset dengan filter login dan shop_id
def dataset():
    if "user_id" not in session:
        flash("Silakan login terlebih dahulu.", "error")
        return redirect(url_for("main.login_page"))

    shop_id = request.args.get("shop_id")
    shop_ids = Review.query.with_entities(Review.shop_id).distinct().all()
    shop_ids = [sid[0] for sid in shop_ids]

    if shop_id:
        reviews = Review.query.filter_by(shop_id=shop_id).all()
    else:
        reviews = Review.query.all()

    return render_template(
        "pages/admin/dataset.html",
        shop_ids=shop_ids,
        selected_shop_id=shop_id,
        reviews=reviews
    )

# ✅ Tambah review via AJAX
def add_review():
    shop_id = request.form.get("shop_id")
    file = request.files.get("file")

    if not shop_id or not file:
        return jsonify(success=False, message="ID Toko dan File wajib diisi."), 400

    # Gunakan timezone Jakarta
    wib = pytz.timezone("Asia/Jakarta")
    now = datetime.now(wib)

    review = Review(
        shop_id=shop_id,
        file=file.filename,
        file_data=file.read(),
        createdAt=now,
        updatedAt=now
    )

    try:
        db.session.add(review)
        db.session.commit()
        return jsonify(success=True, message="Review berhasil ditambahkan!")
    except Exception as e:
        db.session.rollback()
        return jsonify(success=False, message=f"Gagal menambahkan review: {e}"), 500
    
# ✅ Edit review via AJAX
def edit_review(id):
    review = Review.query.get_or_404(id)
    shop_id = request.form.get("shop_id")
    file = request.files.get("file")

    if shop_id:
        review.shop_id = shop_id
    if file:
        review.file = file.filename
        review.file_data = file.read()

    wib = pytz.timezone('Asia/Jakarta')
    review.updatedAt = datetime.now(wib)

    try:
        db.session.commit()
        return jsonify(success=True, message="✅ Review berhasil diperbarui!")
    except Exception as e:
        db.session.rollback()
        return jsonify(success=False, message=f"Gagal update: {e}")

# ✅ Hapus review via AJAX
def delete_review(id):
    review = Review.query.get_or_404(id)

    try:
        db.session.delete(review)
        db.session.commit()
        return jsonify(success=True, message="✅ Review berhasil dihapus!")
    except Exception as e:
        db.session.rollback()
        return jsonify(success=False, message=f"Gagal menghapus review: {e}")