from datetime import datetime
from app import db

class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('review.id'), nullable=False)

    # Hasil Analisis
    total_all = db.Column(db.Integer, nullable=True)
    total_pos = db.Column(db.Integer, nullable=True)
    total_neg = db.Column(db.Integer, nullable=True)
    persen_pos = db.Column(db.Float, nullable=True)
    toko_label = db.Column(db.String(255), nullable=True)
    aspek = db.Column(db.String(255), nullable=True)
    persen_tertinggi = db.Column(db.Float, nullable=True)

    createdAt = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction {self.id} - Review {self.review_id}>"
