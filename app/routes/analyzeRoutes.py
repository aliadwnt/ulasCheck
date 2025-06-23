from flask import Blueprint
from app.controllers.analyzeController import (
    analyze_review,
    download_review,
    cancel_review
)

# Blueprint untuk grup route /analyze/*
analyze = Blueprint("analyze", __name__, url_prefix="/analyze")

# Route: Proses analisis review berdasarkan ID
@analyze.route("/<int:review_id>", methods=["POST"])
def analyze_route(review_id):
    return analyze_review(review_id)

# Route: Unduh file hasil scraping review
@analyze.route("/download/<int:review_id>", methods=["POST"])
def download_route(review_id):
    return download_review(review_id)

# Route: Batalkan dan hapus file review
@analyze.route("/cancel/<int:review_id>", methods=["GET"])
def cancel_route(review_id):
    return cancel_review(review_id)

