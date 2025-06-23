from app import db
from datetime import datetime

class Evaluation(db.Model):
    __tablename__ = 'evaluation'

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    f1_score = db.Column(db.Float, nullable=False)
    training_time = db.Column(db.Float, nullable=True)
    total_review = db.Column(db.Integer, nullable=True)
    positif = db.Column(db.Integer, nullable=True)
    negatif = db.Column(db.Integer, nullable=True)
    persen_positif = db.Column(db.Float, nullable=True)
    label_toko = db.Column(db.String(50), nullable=True)
    aspek_tertinggi = db.Column(db.String(100), nullable=True)
    jumlah_aspek = db.Column(db.Integer, nullable=True)
    persen_aspek = db.Column(db.Float, nullable=True)
    evaluationAt = db.Column(db.DateTime, default=datetime.utcnow)