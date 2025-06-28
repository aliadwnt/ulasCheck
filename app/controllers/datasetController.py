from flask import request, render_template, redirect, url_for, flash, jsonify
from app import db
from app.models.datasetModel import Dataset
from datetime import datetime
import csv
import re

def remove_emoji(text):
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"  # Emoticon
        "\U0001F300-\U0001F5FF"  # Simbol & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Bendera
        "\U00002700-\U000027BF"  # Simbol tambahan
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

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

# ==========================
# UPLOAD DATASET CSV
# ==========================
def upload_dataset():
    file = request.files.get('file')
    if not file:
        return jsonify({"message": "File tidak ditemukan."}), 400

    if not file.filename.lower().endswith('.csv'):
        return jsonify({"message": "Pastikan file berformat .csv"}), 400

    try:
        stream = file.stream.read().decode('utf-8').splitlines()
        csv_reader = csv.DictReader(stream)

        # Buat header di CSV jadi lowercase semua
        csv_reader.fieldnames = [header.strip().lower() for header in csv_reader.fieldnames]

        print(f"Header terbaca: {csv_reader.fieldnames}")

        required_fields = ['username', 'produk', 'review', 'rating', 'reviewat']
        if not all(field in csv_reader.fieldnames for field in required_fields):
            return jsonify({"message": "Format header CSV tidak sesuai."}), 400

        data_inserted = 0
        for row in csv_reader:
            try:
                # Ubah key row jadi lowercase semua
                row = {key.lower(): value for key, value in row.items()}

                # Baca tanggal
                review_date = None
                try:
                    review_date = datetime.strptime(row['reviewat'], '%m/%d/%Y %H:%M')
                except ValueError:
                    try:
                        review_date = datetime.strptime(row['reviewat'], '%d/%m/%Y %H:%M')
                    except ValueError:
                        try:
                            review_date = datetime.strptime(row['reviewat'], '%Y-%m-%d %H:%M')
                        except ValueError:
                            try:
                                review_date = datetime.strptime(row['reviewat'], '%Y-%m-%d')
                            except ValueError:
                                print(f"Format tanggal tidak valid pada baris: {row['reviewat']}")
                                continue  # Skip baris yang error

                new_data = Dataset(
                    username=row['username'],
                    product=remove_emoji(row['produk']),
                    review=remove_emoji(row['review']),
                    rating=int(row['rating']),
                    reviewAt=review_date
                )
                db.session.add(new_data)
                data_inserted += 1

            except Exception as e:
                print(f"Error pada baris: {row}, Error: {e}")
                continue

        db.session.commit()
        return jsonify({"message": f"Upload dataset berhasil. Total data masuk: {data_inserted}"}), 200

    except Exception as e:
        db.session.rollback()
        print(e)
        return jsonify({"message": "Gagal memproses file. Pastikan data valid."}), 500