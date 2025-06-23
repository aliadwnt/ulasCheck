from app import db
from datetime import datetime

class ModelEvaluation(db.Model):
    __tablename__ = 'evaluation'

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    training_time = db.Column(db.Float)
    total_review = db.Column(db.Integer)
    positif = db.Column(db.Integer)
    negatif = db.Column(db.Integer)
    persen_positif = db.Column(db.Float)
    label_toko = db.Column(db.String(100))
    aspek_tertinggi = db.Column(db.String(100))
    jumlah_aspek = db.Column(db.Integer)
    persen_aspek = db.Column(db.Float)
    evaluationAt = db.Column(db.DateTime, default=datetime.utcnow)
