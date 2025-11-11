import pickle
from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb

with open('Insurance-premium-prediction-model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('predict_premium')

@app.route('/predict_premium', methods=['POST'])
def predict_premium():

    customer = request.get_json()

    if customer is None:
        return jsonify({"error": "Invalid or missing JSON"}), 400

    X = dv.transform([customer])

    dmatrix = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))

    pred = model.predict(dmatrix)[0]

    return jsonify({
        "insurance_premium_prediction": float(pred)
    })