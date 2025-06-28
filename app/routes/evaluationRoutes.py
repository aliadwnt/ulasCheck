from flask import Blueprint
from app.controllers.evaluationController import evaluate_model, show_evaluation, get_evaluation, delete_evaluation


evaluation = Blueprint("evaluation", __name__, url_prefix="/admin/evaluation")

evaluation.route("/start", methods=["POST"])(evaluate_model)
evaluation.route("/", methods=["GET"])(show_evaluation)
evaluation.route('/<int:id>', methods=['GET'])(get_evaluation)
evaluation.route('/delete/<int:id>', methods=['POST'])(delete_evaluation)