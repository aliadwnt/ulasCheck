import json
import os
from flask import jsonify, render_template
from app import db
from app.models.evaluationModel import ModelEvaluation

# Fungsi render halaman evaluasi (template HTML)
def render_evaluation_page():
    return render_template("pages/admin/evaluation.html")

# Fungsi pemrosesan model (dummy / placeholder, sesuaikan dengan real logic-mu)
def train_model_controller():
    # Dummy result (ganti sesuai hasil pelatihan sebenarnya)
    dummy_result = [{
        "model": "SVM",
        "accuracy": 0.89,
        "precision": 0.85,
        "recall": 0.87,
        "f1_score": 0.86,
        "time_process": 2.34
    }]
    return jsonify({"results": dummy_result})

# Fungsi menyimpan evaluasi ke database
def save_evaluation_to_db():
    path = "model/evaluation_summary.json"
    if not os.path.exists(path):
        return jsonify({"error": "evaluation_summary.json tidak ditemukan"}), 404

    try:
        with open(path, "r") as file:
            data = json.load(file)
    except Exception as e:
        return jsonify({"error": f"Gagal membaca file JSON: {str(e)}"}), 500

    try:
        new_eval = ModelEvaluation(
            model_name=data.get("model_name"),
            accuracy=float(data.get("accuracy", 0)),
            precision=float(data.get("precision", 0)),
            recall=float(data.get("recall", 0)),
            f1_score=float(data.get("f1_score", 0)),
            training_time=float(data.get("training_time", 0)),
            total_review=int(data.get("total_review", 0)),
            positif=int(data.get("positif", 0)),
            negatif=int(data.get("negatif", 0)),
            persen_positif=float(data.get("persen_positif", 0)),
            label_toko=data.get("label_toko", ""),
            aspek_tertinggi=data.get("aspek_tertinggi", ""),
            jumlah_aspek=int(data.get("jumlah_aspek", 0)),
            persen_aspek=float(data.get("persen_aspek", 0))
        )

        db.session.add(new_eval)
        db.session.commit()
        return jsonify({"message": "✅ Evaluasi berhasil disimpan ke database!"})

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Gagal menyimpan ke database: {str(e)}"}), 500
