from flask import render_template, flash, redirect,send_file, request, session, url_for
from app.models.reviewModel import Review
import io

def show_reviews():
    reviews = Review.query.order_by(Review.created_at.desc()).all()
    return render_template("pages/admin/history.html", reviews=reviews)

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