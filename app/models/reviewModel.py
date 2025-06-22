from datetime import datetime
from app import db

class Review(db.Model):
    __tablename__ = "review"
    id = db.Column(db.Integer, primary_key=True)
    shop_id = db.Column(db.String(255))
    file = db.Column(db.String(255))
    file_data = db.Column(db.LargeBinary)
    createdAt = db.Column(db.DateTime, default=datetime.utcnow)
    updatedAt = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)