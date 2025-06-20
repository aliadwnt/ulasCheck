from app import db

class Dataset(db.Model):
    __tablename__ = "dataset"
    id = db.Column(db.Integer, primary_key=True)
    original_review = db.Column(db.Text, nullable=False)
    cleaned_review = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=db.func.now())

    review_id = db.Column(db.Integer, db.ForeignKey("review.id"), nullable=False)