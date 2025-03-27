from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the saved model
save_path = os.path.join(os.getcwd(), "clv_model.pkl")
with open(save_path, "rb") as file:
    loaded_model = pickle.load(file)

print("✅ Model loaded successfully.")

@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from frontend
        data = request.get_json()

        # Validate required fields
        if "purchaseHistory" not in data or "customerEngagement" not in data:
            return jsonify({"error": "Missing required fields"}), 400

        # Prepare input data for the model
        input_data = pd.DataFrame({
            'Total Purchases ($)': [float(data["purchaseHistory"])],
            'Engagement Score': [float(data["customerEngagement"])]
        })

        # Make predictions
        prediction = loaded_model.predict(input_data)

        return jsonify({"predicted_clv": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
