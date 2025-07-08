from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('clv_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract and convert input features
        customer_id = data['customerID']
        purchase_history = float(data['purchaseHistory'])
        customer_engagement = float(data['customerEngagement'])

        # Prepare the data in the shape the model expects (2D array)
        features = np.array([[purchase_history, customer_engagement]])

        # Predict using the trained model
        prediction = model.predict(features)

        return jsonify({'predicted_clv': round(float(prediction[0]), 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
