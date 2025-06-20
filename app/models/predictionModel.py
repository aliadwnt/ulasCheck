from app import db
from datetime import datetime

class PredictionResult(db.Model):
    __tablename__ = 'prediction'
    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('review.id'), nullable=False)
    review = db.relationship('Review', backref=db.backref('predictions', lazy=True))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    total_review = db.Column(db.Integer)
    total_pos = db.Column(db.Integer)
    total_neg = db.Column(db.Integer)
    persen_pos = db.Column(db.Float)
    label_toko = db.Column(db.String(30))    
    aspek_utama = db.Column(db.String(50))
