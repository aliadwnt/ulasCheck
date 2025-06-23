from flask import Blueprint
from app.controllers.evaluationController import evaluate_model, show_evaluation

evaluation = Blueprint("evaluation", __name__, url_prefix="/evaluation")

evaluation.route("/start", methods=["POST"])(evaluate_model)
evaluation.route("/", methods=["GET"])(show_evaluation)
