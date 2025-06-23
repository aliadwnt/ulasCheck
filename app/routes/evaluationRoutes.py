from flask import Blueprint
from app.controllers.evaluationController import (
    render_evaluation_page,
    train_model_controller,
    save_evaluation_to_db
)

evaluation = Blueprint("evaluation", __name__)

@evaluation.route("/admin/evaluation")
def evaluation_page():
    return render_evaluation_page()

@evaluation.route("/train-svm")
def train_svm():
    return train_model_controller()

@evaluation.route("/save-evaluation", methods=["POST"])
def save_evaluation():
    return save_evaluation_to_db()
