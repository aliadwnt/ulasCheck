from flask import request, render_template, redirect, url_for, flash, jsonify 
from app import db
from app.models.datasetModel import Dataset
from datetime import datetime
import csv, re, locale

locale.setlocale(locale.LC_TIME, 'id_ID.UTF-8')

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
                review_date = d.reviewAt.strftime('%d %B %Y %H:%M')
            else:
                review_date = "-"

            if d.createdAt:
                formatted_createdAt = d.createdAt.strftime('%d %B %Y %H:%M')
            else:
                formatted_createdAt = '-'

            dataset.append({
                "id": d.id,
                "username": d.username,
                "product": d.product,
                "review": d.review,
                "rating": d.rating,
                "reviewAt": review_date,
                "createdAt": formatted_createdAt
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
        flash("Data berhasil ditambahkan!", "success")
        return redirect(url_for("dataset.get_all_dataset"))

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
        flash("Data berhasil diperbarui!", "success")
        return redirect(url_for("dataset.get_all_dataset"))

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
        flash("✅ <strong>Sukses!</strong> Data review berhasil dihapus.", "success")
        return redirect(url_for("dataset.get_all_dataset"))
    except Exception as e:
        db.session.rollback()
        flash("❌ Terjadi kesalahan saat menghapus data.", "danger")
        return redirect(url_for("dataset.get_all_dataset"))

# ==========================
# UPLOAD DATASET CSV
# ==========================
def upload_dataset():
    file = request.files.get('file')
    if not file:
        flash("File tidak ditemukan.", "danger")
        return redirect(url_for("dataset.get_all_dataset"))

    if not file.filename.lower().endswith('.csv'):
        flash("Pastikan file berformat .csv", "danger")
        return redirect(url_for("dataset.get_all_dataset"))

    try:
        stream = file.stream.read().decode('utf-8').splitlines()
        csv_reader = csv.DictReader(stream)
        csv_reader.fieldnames = [header.strip().lower() for header in csv_reader.fieldnames]

        required_fields = ['username', 'produk', 'review', 'rating', 'reviewat']
        if not all(field in csv_reader.fieldnames for field in required_fields):
            flash("Format header CSV tidak sesuai.", "danger")
            return redirect(url_for("dataset.get_all_dataset"))

        data_inserted = 0
        for row in csv_reader:
            try:
                row = {key.lower(): value for key, value in row.items()}

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
                                continue

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
                continue

        db.session.commit()
        flash(f"Upload dataset berhasil, Total data masuk: {data_inserted} data.", "success")
        return redirect(url_for("dataset.get_all_dataset"))

    except Exception as e:
        db.session.rollback()
        flash("Gagal memproses file. Pastikan data valid.", "danger")
        return redirect(url_for("dataset.get_all_dataset"))