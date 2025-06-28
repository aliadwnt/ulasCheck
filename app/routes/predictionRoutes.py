from flask import Blueprint
from app.controllers import predictionController

prediction = Blueprint('prediction', __name__)

@prediction.route('/predictions', methods=['POST'])
def create_prediction():
    return predictionController.create_prediction()

@prediction.route('/predictions', methods=['GET'])
def get_all_predictions():
    return predictionController.get_all_predictions()

@prediction.route('/predictions/<int:prediction_id>', methods=['GET'])
def get_prediction_by_id(prediction_id):
    return predictionController.get_prediction_by_id(prediction_id)

@prediction.route('/predictions/<int:prediction_id>', methods=['DELETE'])
def delete_prediction(prediction_id):
    return predictionController.delete_prediction(prediction_id)
