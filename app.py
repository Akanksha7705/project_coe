from flask import Flask, jsonify, request, render_template
import pickle
import os
import pandas as pd

app = Flask(__name__)

# Define the path where the model is saved
save_path = os.path.join(os.getcwd(), "clv_model.pkl")

# Load the saved model once when the app starts
with open(save_path, "rb") as file:
    loaded_model = pickle.load(file)

print("Model loaded successfully.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Invalid input, JSON required'}), 400

        # Extract values and ensure they are floats
        total_purchases = float(data.get('total_purchases', 0))
        engagement_score = float(data.get('engagement_score', 0))

        # Prepare input for the model
        input_data = pd.DataFrame({
            'Total Purchases ($)': [total_purchases],
            'Engagement Score': [engagement_score]
        })

        # Make prediction
        prediction = loaded_model.predict(input_data)

        # Convert NumPy float32 to Python float before returning JSON
        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True)
