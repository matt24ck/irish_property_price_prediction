
# Irish Property Price Prediction — Documentation

## Overview

This project aims to analyze, visualize, and predict property prices in Ireland using publicly available property transaction data. It performs Exploratory Data Analysis (EDA) and trains an XGBoost regression model based on features like sale date, routing key, town, and county.

## Dataset

- Source: `PPR-ALL.csv`
- Key Features:
  - `Date of Sale`: Date on which the property was sold.
  - `Eircode`: Property's Eircode
  - `Address`: Property's Address
  - `Price`: Sale price in euros (target variable)

## Exploratory Data Analysis (EDA)

The EDA section explores the data to identify patterns, outliers, and distributions:

- Missing Values: Summary of null entries
- Sale Trends Over Time: Average price by year
- Top Routing Keys: By mean price
- Boxplots by County: By price

## Model Training

- Model Used: XGBoost Regressor
- Preprocessing:
  - `LabelEncoder` used to encode categorical variables
- Target: `Price`
- Features:
  - `SaleYear`, `SaleMonth`, `routing_key`, `Town`, `County`

### Model Saving

- Model is saved using `pickle` to a file called `xgb_model.pkl`
- Label encoders are saved as a dictionary to `label_encoders.pkl`

## Deployment Overview

- A simple Flask web app is created to host the model.
- Users can enter:
  - Routing Key
  - Sale Year and Month
  - Town and County
- The app predicts a valuation estimate.

## Monitoring and Logging

- Logging implemented using Python’s `logging` module
- Tracks:
  - Prediction inputs
  - Number of requests
- Usage counters saved in logs (`usage.log`)

## Deployment Options

- Render.com (recommended for Flask apps)
- Replit (for demos)
- Hosting includes monitoring user input frequency

## How to Run

### 1. Install dependencies:

```bash
pip install pandas xgboost flask seaborn matplotlib scikit-learn
```

### 2. Run the notebook for EDA & model training

### 3. Start the Flask app:

```bash
python app.py
```

### 4. Visit:

```
http://localhost:5000/
```

## File Structure

```
IrishPropertyPrice/
│
├── EIRCODE-PROPERTIES.csv         # Input dataset
├── model/
│   └── IrishPropertyPrice.ipynb   # Jupyter notebook (EDA + training)
│
├── web_app/
│   └──  xgb_model.pkl             # Trained XGBoost model
│   └──  label_encoders.pkl        # Pickled label encoders
│   └──  app.py                    # Flask application
│   └──  templates/
│        └── index.html            # Web UI for prediction
│   └──  usage.log                 # User input logs
```

## To Do

- Add more features (e.g. building type, area, bedrooms)
- Improve model with hyperparameter tuning
- Track user sessions or add authentication for advanced use
