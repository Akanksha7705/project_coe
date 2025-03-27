import pickle
import numpy as np
from flask import Flask, request, jsonify 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows frontend requests from different origins
CORS(app, resources={r"/predict": {"origins": "*"}})

with open("clv_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Flask Backend is Running!"

@app.route('/predict', methods=['POST'])
def predict_clv():
    try:
        print("Received request:", request.method)  # Debugging print
        data = request.get_json()
        print("Received data:", data)  # Debugging print
        
        if not data:
            return jsonify({"error": "Empty request"}), 400

        customer_id = data.get("customerID", "Unknown")
        purchase_history = float(data.get("purchaseHistory", 0))
        customer_engagement = float(data.get("customerEngagement", 0))

        input_data = np.array([[purchase_history, customer_engagement]])
        predicted_clv = model.predict(input_data)[0]

        return jsonify({"customer_id": customer_id, "predicted_clv": round(predicted_clv, 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)
