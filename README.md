# 🍷 Wine Quality Prediction Flask App

A Flask web application that predicts wine quality (Low / Medium / High) using a trained machine learning model. The app uses features from physicochemical tests to classify the quality of red wine.

---
## 🚀 Live Demo
https://wine-quality-flask.onrender.com/

## 🧠 Model Overview

- **Algorithm**: Random Forest (default version was the best-performing)
- **Preprocessing**:
  - Outlier removal using IQR
  - Standardization using `StandardScaler`
  - Label encoding for the quality target variable
- **Classes**: `Low`, `Medium`, `High`

# 📁 Project Structure

wine_quality_flask/
│
├── app.py # Main Flask app
├── Procfile # For deployment (Render/Heroku)
├── requirements.txt # Python dependencies
├── templates/
│ └── index.html # Web form for user input
├── scaler.pkl # Saved StandardScaler
├── label_encoder.pkl # Saved LabelEncoder
├── best_random_forest_model.pkl # Trained ML model
└── README.md # You are here

## ⚙️ Installation & Setup (Local)

### 🔹 Prerequisites

- Python 3.x
- pip
- Git

### 🔹 Clone the Repository

git clone https://github.com/skilfulkay/wine_quality_flask.git
cd wine_quality_flask

 Install Dependencies
pip install -r requirements.txt

🔹 Run the App

python app.py
Visit http://127.0.0.1:5000 in your browser.
