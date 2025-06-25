from datetime import datetime
from app import db
from sqlalchemy.dialects.mysql import LONGBLOB

class Review(db.Model):
    __tablename__ = "review"

    id = db.Column(db.Integer, primary_key=True)
    shop_id = db.Column(db.String(255))
    shop_name = db.Column(db.String(255))
    file = db.Column(db.String(255))
    file_data = db.Column(LONGBLOB)
    createdAt = db.Column(db.DateTime, default=datetime.utcnow)
    updatedAt = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)