from app import db

class Dataset(db.Model):
    __tablename__ = "dataset"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    product = db.Column(db.String(255), nullable=False)
    review = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    reviewAt = db.Column(db.DateTime, nullable=False)
    createdAt = db.Column(db.DateTime, default=db.func.now())