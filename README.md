
# Project COE

This project is a web-based platform designed for predicting **Customer Lifetime Value (CLV)** using machine learning. It combines a user-friendly frontend with a Flask backend that handles model predictions and user interactions.

## 🔍 Features

- 🧠 **Machine Learning Model**: Predicts customer lifetime value using a `.pkl` model (`clv_model.pkl`).
- 🧾 **Customer Data Input**: Allows users to input customer data for prediction.
- 📊 **Data Handling**: Accepts and processes a CSV file (`customer_clv_data.csv`) containing historical customer data.
- 🌐 **Frontend Interface**: HTML pages for `Home`, `Login`, `About Us`, and `Contact Us`.
- 🔐 **Login Functionality**: Simple login system to restrict access to the prediction interface.
- 🔁 **Integrated Workflow**: HTML + Flask backend + Python model integration.

## 🗂️ Project Structure

```
project_coe/
├── app.html              # Frontend page for input
├── app.py                # Flask app file
├── backend.py            # Handles prediction and routing
├── clv_model.pkl         # Pre-trained machine learning model
├── customer_clv_data.csv # Customer data for analysis
├── aboutus.html          # About Us page
├── contactus.html        # Contact page
├── home.html             # Main dashboard/home page
├── login.html            # Login page
├── index (1).html        # CLV input and prediction form
```

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Akanksha7705/project_coe.git
   ```

2. Install the required packages:
   ```bash
   pip install flask pandas scikit-learn
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open your browser and go to:
   ```
   http://localhost:5000
   ```

## 📌 Notes

- Make sure `clv_model.pkl` and `customer_clv_data.csv` are present in the root directory.
- You can extend the model to use real-time data or connect it to a database in the future.

## 📫 Contact

For any queries or feedback, please reach out via the **Contact Us** page in the app.
