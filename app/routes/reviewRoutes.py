from flask import Blueprint
from app.controllers import reviewController

review = Blueprint("review", __name__)

#halaman history
review.add_url_rule("/admin/history", view_func=reviewController.show_reviews)
review.add_url_rule('/admin/history/download/<int:id>',view_func=reviewController.download_file)

#halaman dataset
review.add_url_rule("/admin/dataset", view_func=reviewController.dataset, methods=["GET"])
review.add_url_rule("/admin/history/download/<int:id>", view_func=reviewController.download_file)
