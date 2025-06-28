from datetime import datetime
from app import db

class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('review.id'), nullable=False)

    total_all = db.Column(db.Integer)
    total_pos = db.Column(db.Integer)
    total_neg = db.Column(db.Integer)
    persen_pos = db.Column(db.Float)
    toko_label = db.Column(db.String(255))
    aspek = db.Column(db.String(255))
    persen_tertinggi = db.Column(db.Float)

    createdAt = db.Column(db.DateTime, default=datetime.utcnow)