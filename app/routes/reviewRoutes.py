from flask import Blueprint
from app.controllers import reviewController

# Membuat Blueprint
review = Blueprint("review", __name__)

# Halaman riwayat review
review.add_url_rule(
    "/admin/history",
    view_func=reviewController.show_reviews,
    methods=["GET"]
)

# Download file dari riwayat review berdasarkan ID
review.add_url_rule(
    "/admin/history/download/<int:id>",
    view_func=reviewController.download_file,
    methods=["GET"]
)

# Tambah review (POST via AJAX, GET sebaiknya tidak perlu)
review.add_url_rule(
    "/admin/review/add",
    view_func=reviewController.add_review,
    methods=["POST"]
)

# Edit review (POST via AJAX, GET sebaiknya tidak perlu)
review.add_url_rule(
    "/admin/review/edit/<int:id>",
    view_func=reviewController.edit_review,
    methods=["POST"]
)

# Hapus review
review.add_url_rule(
    "/admin/review/delete/<int:id>",
    view_func=reviewController.delete_review,
    methods=["POST"]
)

# Halaman dataset
review.add_url_rule(
    "/admin/dataset",
    view_func=reviewController.dataset,
    methods=["GET"]
)
