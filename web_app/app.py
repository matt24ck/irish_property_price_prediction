from flask import Flask, request, render_template, jsonify
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pickle
import logging


logging.basicConfig(
    filename='usage.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

app = Flask(__name__)

# Load your trained model and label encoders (assumed saved with pickle)
model = pickle.load(open('xgb_model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

categorical_cols = ['routing_key']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Extract inputs from form
    sale_year = int(data['SaleYear'])
    sale_month = int(data['SaleMonth'])
    routing_key = data['routing_key'].strip().upper()

    logging.info(f"Prediction requested | Year: {sale_year}, Month: {sale_month}, Routing Key: {routing_key}")


    # Create DataFrame for model input
    df = pd.DataFrame({
        'SaleYear': [sale_year],
        'SaleMonth': [sale_month],
        'routing_key': [routing_key],
    })

    # Encode categorical columns with existing encoders
    for col in categorical_cols:
        le = label_encoders[col]
        try:
            df[f'{col}_encoded'] = le.transform(df[col])
        except ValueError:
            # Handle unseen labels gracefully
            df[col] = 0  # or some default value
    df = df.drop('routing_key', axis=1)
    # Predict
    pred = model.predict(df)[0]

    return render_template('index.html', prediction=f"Estimated Price: €{pred:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)