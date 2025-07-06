from flask import render_template, redirect, url_for, flash, jsonify
from app.models.datasetModel import Dataset
from app.models.evaluationModel import Evaluation
from app.analyze import clean_text, vectorizer, model
from app.extensions import db

import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time


# Tampilkan halaman evaluasi
def show_evaluation():
    logs = Evaluation.query.order_by(Evaluation.evaluationAt.desc()).all()
    return render_template('pages/admin/evaluation.html', logs=logs)

def start_evaluate():
    data = Dataset.query.all()
    if not data:
        flash("Tidak ada data untuk evaluasi.", "danger")
        return redirect(url_for('dataset.show_dataset'))

    start_time = time.time()

    df = pd.DataFrame([{
        "review": d.review,
        "rating": d.rating
    } for d in data])

    df["clean"] = df["review"].apply(clean_text)
    df.dropna(subset=["clean", "rating"], inplace=True)

    X = vectorizer.transform(df["clean"])
    y_true = (df["rating"] >= 4).astype(int)
    y_pred = model.predict(X)

    acc = round(accuracy_score(y_true, y_pred), 4)
    prec = round(precision_score(y_true, y_pred), 4)
    rec = round(recall_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred), 4)

    duration = round(time.time() - start_time, 2)

    log = Evaluation(
        model_name="SVM",
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        training_time=duration,
        total_review=len(df),
        positif=int((y_true == 1).sum()),
        negatif=int((y_true == 0).sum()),
        persen_positif=round((y_true == 1).mean() * 100, 2),
        label_toko="Toko Direkomendasikan" if (y_true == 1).mean() >= 0.5 else "Toko Tidak Direkomendasikan",
        evaluationAt=datetime.now()
    )

    db.session.add(log)
    db.session.commit()

    flash("Evaluasi berhasil dilakukan.", "success")
    return redirect(url_for('evaluation.show_evaluation'))

# Evaluasi model
def evaluate_model():
    data = Dataset.query.all()
    if not data:
        flash("Tidak ada data untuk evaluasi.", "danger")
        return redirect(url_for('dataset.show_dataset'))

    start_time = time.time()

    df = pd.DataFrame([{
        "review": d.review,
        "rating": d.rating
    } for d in data])

    df["clean"] = df["review"].apply(clean_text)
    df.dropna(subset=["clean", "rating"], inplace=True)

    X = vectorizer.transform(df["clean"])
    y_true = (df["rating"] >= 4).astype(int)
    y_pred = model.predict(X)

    acc = round(accuracy_score(y_true, y_pred), 4)
    prec = round(precision_score(y_true, y_pred), 4)
    rec = round(recall_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred), 4)

    duration = round(time.time() - start_time, 2)

    log = Evaluation(
        model_name="SVM",
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        training_time=duration,
        total_review=len(df),
        positif=int((y_true == 1).sum()),
        negatif=int((y_true == 0).sum()),
        persen_positif=round((y_true == 1).mean() * 100, 2),
        label_toko="Toko Direkomendasikan" if (y_true == 1).mean() >= 0.5 else "Toko Tidak Direkomendasikan",
        evaluationAt=datetime.now()
    )

    db.session.add(log)
    db.session.commit()

    flash("Evaluasi berhasil dilakukan.", "success")
    return redirect(url_for('evaluation.show_evaluation'))


# Ambil data evaluasi berdasarkan ID
def get_evaluation(id):
    evaluation = Evaluation.query.get_or_404(id)
    return jsonify(evaluation.to_dict())


def delete_evaluation(id):
    try:
        evaluation = Evaluation.query.get(id)
        if not evaluation:
            flash("Data evaluasi tidak ditemukan.", "error")
            return redirect(url_for("evaluation.show_evaluation"))  # Fix nama endpoint

        db.session.delete(evaluation)
        db.session.commit()
        flash("Data evaluasi berhasil dihapus.", "success")
        return redirect(url_for("evaluation.show_evaluation"))  # Fix nama endpoint

    except Exception as e:
        db.session.rollback()
        flash(f"Terjadi kesalahan: {str(e)}", "error")
        return redirect(url_for("evaluation.show_evaluation"))  # Fix nama endpoint
