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
    evaluationAt = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "training_time": self.training_time,
            "total_review": self.total_review,
            "positif": self.positif,
            "negatif": self.negatif,
            "persen_positif": self.persen_positif,
            "label_toko": self.label_toko,
            "evaluationAt": self.evaluationAt.strftime('%Y-%m-%d %H:%M:%S')
        }
