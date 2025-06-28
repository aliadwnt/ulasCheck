from flask import request, jsonify
from app import db
from app.models.predictionModel import Prediction

# Create Prediction
def create_prediction():
    data = request.json
    try:
        new_prediction = Prediction(
            review_id=data['review_id'],
            total_all=data.get('total_all'),
            total_pos=data.get('total_pos'),
            total_neg=data.get('total_neg'),
            persen_pos=data.get('persen_pos'),
            toko_label=data.get('toko_label'),
            aspek=data.get('aspek'),
            persen_tertinggi=data.get('persen_tertinggi')
        )
        db.session.add(new_prediction)
        db.session.commit()
        return jsonify({'message': 'Prediction created successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Get All Predictions
def get_all_predictions():
    predictions = Prediction.query.all()
    result = []
    for pred in predictions:
        result.append({
            'id': pred.id,
            'review_id': pred.review_id,
            'total_all': pred.total_all,
            'total_pos': pred.total_pos,
            'total_neg': pred.total_neg,
            'persen_pos': pred.persen_pos,
            'toko_label': pred.toko_label,
            'aspek': pred.aspek,
            'persen_tertinggi': pred.persen_tertinggi,
            'createdAt': pred.createdAt
        })
    return jsonify(result)

# Get Prediction by ID
def get_prediction_by_id(prediction_id):
    prediction = Prediction.query.get(prediction_id)
    if not prediction:
        return jsonify({'message': 'Prediction not found'}), 404

    result = {
        'id': prediction.id,
        'review_id': prediction.review_id,
        'total_all': prediction.total_all,
        'total_pos': prediction.total_pos,
        'total_neg': prediction.total_neg,
        'persen_pos': prediction.persen_pos,
        'toko_label': prediction.toko_label,
        'aspek': prediction.aspek,
        'persen_tertinggi': prediction.persen_tertinggi,
        'createdAt': prediction.createdAt
    }
    return jsonify(result)

# Delete Prediction
def delete_prediction(prediction_id):
    prediction = Prediction.query.get(prediction_id)
    if not prediction:
        return jsonify({'message': 'Prediction not found'}), 404

    db.session.delete(prediction)
    db.session.commit()
    return jsonify({'message': 'Prediction deleted successfully'})
