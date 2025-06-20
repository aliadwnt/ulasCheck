from app import db

class Review(db.Model):
    __tablename__ = "review"
    id = db.Column(db.Integer, primary_key=True)
    shop_id = db.Column(db.String(255))
    file = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=db.func.now())

    datasets = db.relationship("Dataset", backref="review", cascade="all, delete-orphan", lazy=True)