from flask import Blueprint
from app.controllers import datasetController

dataset = Blueprint("dataset", __name__, url_prefix="/admin/dataset")

# Route untuk menampilkan semua dataset
@dataset.route("/", methods=["GET"])
def get_all_dataset():
    return datasetController.get_all_dataset()

# Route untuk menambah dataset
@dataset.route("/add", methods=["POST"])
def add_dataset():
    return datasetController.add_dataset()

# Route untuk mengedit dataset
@dataset.route("/edit/<int:id>", methods=["POST"])
def edit_dataset(id):
    return datasetController.edit_dataset(id)

# Route untuk menghapus dataset
@dataset.route("/delete/<int:id>", methods=["POST"])
def delete_dataset(id):
    return datasetController.delete_dataset(id)

# Route untuk upload dataset CSV
@dataset.route("/upload", methods=["POST"])
def upload_dataset():
    return datasetController.upload_dataset()
