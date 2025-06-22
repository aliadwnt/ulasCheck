from flask import Blueprint
from app.controllers import reviewController

review = Blueprint("review", __name__)

# ✅ HISTORY PAGE (Riwayat)
review.add_url_rule(
    "/admin/history",
    view_func=reviewController.show_reviews,
    methods=["GET"]
)
review.add_url_rule(
    "/admin/history/download/<int:id>",
    view_func=reviewController.download_file,
    methods=["GET"]
)

# ✅ DATASET PAGE
review.add_url_rule(
    "/admin/dataset",
    view_func=reviewController.dataset,
    methods=["GET"]
)

# ✅ CRUD (Form manual atau AJAX bisa disesuaikan)
review.add_url_rule(
    "/admin/review/add",
    view_func=reviewController.add_review,
    methods=["GET", "POST"]
)
review.add_url_rule(
    "/admin/review/edit/<int:id>",
    view_func=reviewController.edit_review,
    methods=["GET", "POST"]
)
review.add_url_rule(
    "/admin/review/delete/<int:id>",
    view_func=reviewController.delete_review,
    methods=["POST"]
)