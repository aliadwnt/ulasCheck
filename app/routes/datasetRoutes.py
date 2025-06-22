from flask import Blueprint
from app.controllers import datasetController

dataset = Blueprint("dataset", __name__, url_prefix="/admin/dataset")

dataset.route("/", methods=["GET"])(datasetController.get_all_dataset)
dataset.route("/add", methods=["POST"])(datasetController.add_dataset)
dataset.route("/edit/<int:id>", methods=["POST"])(datasetController.edit_dataset)
dataset.route("/delete/<int:id>", methods=["POST"])(datasetController.delete_dataset)
