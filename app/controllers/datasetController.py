from flask import request, jsonify, render_template
from app import db
from app.models.datasetModel import Dataset
from datetime import datetime

# ==========================
# GET ALL DATASET
# ==========================
def get_all_dataset():
    try:
        data = Dataset.query.order_by(Dataset.id.desc()).all()
        dataset = []

        for d in data:
            if isinstance(d.reviewAt, datetime):
                review_date = d.reviewAt.strftime('%Y-%m-%d')
            else:
                review_date = "-"

            dataset.append({
                "id": d.id,
                "username": d.username,
                "product": d.product,
                "review": d.review,
                "rating": d.rating,
                "reviewAt": review_date
            })

        return render_template("pages/admin/dataset.html", dataset=dataset)

    except Exception as e:
        return f"Error: {str(e)}", 500


# ==========================
# ADD DATASET
# ==========================
def add_dataset():
    try:
        username = request.form.get("username")
        product = request.form.get("product")
        review = request.form.get("review")
        rating = request.form.get("rating")
        reviewAt = request.form.get("reviewAt")

        if not all([username, product, review, rating, reviewAt]):
            return "Lengkapi semua field!", 400

        try:
            reviewAt_dt = datetime.strptime(reviewAt, "%Y-%m-%d")
        except ValueError:
            return "Format tanggal tidak valid (harus YYYY-MM-DD)", 400

        new_data = Dataset(
            username=username,
            product=product,
            review=review,
            rating=rating,
            reviewAt=reviewAt_dt
        )
        db.session.add(new_data)
        db.session.commit()
        return "success"

    except Exception as e:
        db.session.rollback()
        return f"Error: {str(e)}", 500


# ==========================
# EDIT DATASET
# ==========================
def edit_dataset(id):
    try:
        data = Dataset.query.get_or_404(id)

        username = request.form.get("username")
        product = request.form.get("product")
        review = request.form.get("review")
        rating = request.form.get("rating")
        reviewAt = request.form.get("reviewAt")

        if not all([username, product, review, rating, reviewAt]):
            return "Lengkapi semua field!", 400

        try:
            reviewAt_dt = datetime.strptime(reviewAt, "%Y-%m-%d")
        except ValueError:
            return "Format tanggal tidak valid", 400

        data.username = username
        data.product = product
        data.review = review
        data.rating = rating
        data.reviewAt = reviewAt_dt

        db.session.commit()
        return "success"

    except Exception as e:
        db.session.rollback()
        return f"Error: {str(e)}", 500


# ==========================
# DELETE DATASET
# ==========================
def delete_dataset(id):
    try:
        data = Dataset.query.get_or_404(id)
        db.session.delete(data)
        db.session.commit()
        return "success"
    except Exception as e:
        db.session.rollback()
        return f"Error: {str(e)}", 500
