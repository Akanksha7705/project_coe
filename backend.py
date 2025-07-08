from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Allow requests from HTML running on a different port (like 5500)

model = joblib.load("clv_model.pkl")  # Load your trained model here

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    customer_id = data.get('customerID')
    purchase_amount = data.get('purchaseAmount')
    engagement_score = data.get('engagementScore')

    if None in (purchase_amount, engagement_score):
        return jsonify({'error': 'Missing input values'}), 400

    try:
        prediction = model.predict([[purchase_amount, engagement_score]])
        return jsonify({'predicted_clv': round(prediction[0], 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
