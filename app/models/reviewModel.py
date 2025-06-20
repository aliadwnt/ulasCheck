from app.models import db
from datetime import datetime

class Review(db.Model):
    __tablename__ = "review"
    id = db.Column(db.Integer, primary_key=True)
    shop_id = db.Column(db.String(50), nullable=False)
    file = db.Column(db.String(255), nullable=False) 
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  