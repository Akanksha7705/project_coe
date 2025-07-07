from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'secret-key'

# Temporary in-memory data stores
users = {}
predictions = {}

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_post():
    username = request.form['username']
    password = generate_password_hash(request.form['password'])

    if username in users:
        return 'User already exists!'
    users[username] = password
    predictions[username] = []  # Init empty predictions
    return redirect(url_for('login'))

@app.route('/login', methods=['POST'])
def login_post():
    username = request.form['username']
    password = request.form['password']

    if username in users and check_password_hash(users[username], password):
        session['username'] = username
        return redirect(url_for('home'))
    return 'Invalid credentials!'

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))

    user = session['username']
    user_predictions = predictions.get(user, [])
    return render_template('home.html', predictions=user_predictions)

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    customer_id = request.form['customer_id']
    result = request.form['prediction']

    predictions[session['username']].append({
        'customer_id': customer_id,
        'result': result
    })
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
