# 📊 Customer Churn Prediction

An end-to-end machine learning application that predicts whether a telecom customer is likely to churn. The project includes data preprocessing, exploratory data analysis, feature engineering, model training and evaluation, SHAP-based explainability, a FastAPI prediction API, and an interactive Streamlit dashboard.

## 🚀 Live Demo

* 🌐 **Streamlit Dashboard:** `https://customer-churn-prediction-brqucwgshappgvlmylftrk4.streamlit.app/`
* 🔌 **FastAPI API:** https://customer-churn-prediction-tn0a.onrender.com
* 📚 **API Documentation:** https://customer-churn-prediction-tn0a.onrender.com/docs

## 🎯 Project Objective

Customer churn is an important problem for subscription-based businesses. Identifying customers who are likely to leave can help businesses take proactive retention measures.

This project uses machine learning to:

* Predict customer churn
* Estimate the probability of churn
* Classify customers according to churn risk
* Explain individual predictions using SHAP
* Provide predictions through a REST API
* Provide an interactive web dashboard

## 🧠 Machine Learning Workflow

The project follows an end-to-end machine learning workflow:

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Model Training
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
SHAP Explainability
     ↓
FastAPI
     ↓
Streamlit Dashboard
```

## 📈 Model Performance

The final model was evaluated on the test dataset.

| Metric    |  Score |
| --------- | -----: |
| Accuracy  |    97% |
| Precision |    97% |
| Recall    |    90% |
| F1 Score  |    93% |
| ROC-AUC   | 0.9932 |

The ROC-AUC score of **0.9932** indicates excellent discrimination between churn and non-churn customers on the evaluation data.

## 🔍 Model Explainability

SHAP (SHapley Additive exPlanations) is used to explain model predictions.

The application provides:

* Global feature importance
* Local explanations for individual customers
* Features increasing churn risk
* Features decreasing churn risk
* SHAP values for the most influential features

This makes the prediction system more interpretable instead of providing only a churn/non-churn result.

## 🖥️ Streamlit Dashboard

The Streamlit dashboard provides an interactive interface for entering customer information and receiving predictions.

### Dashboard features

* 📈 Model performance metrics
* 📊 Confusion matrix
* 📉 ROC curve
* 👤 Customer information input
* 🔮 Churn prediction
* 🎯 Churn probability
* ⚠️ Churn risk classification
* 💡 SHAP-based prediction explanation
* 📊 Feature impact visualization

## 🔌 FastAPI

The machine learning model is exposed through a FastAPI backend.

### Main endpoints

#### Health Check

```text
GET /health
```

Returns the API health status and confirms that the model is loaded.

#### Prediction

```text
POST /predict
```

Accepts customer information and returns:

* Churn prediction
* Churn probability
* SHAP explanation

### API Documentation

Interactive Swagger documentation is available at:

https://customer-churn-prediction-tn0a.onrender.com/docs

## ☁️ Deployment

The project uses separate services for the backend and frontend.

### Backend

The FastAPI application is deployed using Render.

```text
Streamlit Dashboard
        ↓
Render FastAPI
        ↓
Trained ML Model
        ↓
Prediction + SHAP Explanation
```

### Frontend

The Streamlit dashboard is deployed using Streamlit Community Cloud.

Users can access the dashboard through a web browser without running the application locally.

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── processed/
│   ├── telco.csv
│   └── telco_clean.csv
│
├── models/
│   ├── best_model.pkl
│   ├── churn_prediction_model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 08_model_explainability.ipynb
│
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── shap_feature_importance.png
│   ├── local_shap_feature_importance.png
│   ├── predictions.csv
│   └── shap_feature_importance.csv
│
├── src/
│   ├── evaluate.ipynb
│   ├── feature_engineering.ipynb
│   ├── pre_processing.ipynb
│   ├── predict.ipynb
│   ├── train.ipynb
│   └── utils.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost
* LightGBM
* XGBoost
* SHAP
* Matplotlib
* Seaborn
* FastAPI
* Pydantic
* Uvicorn
* Streamlit
* Requests
* Joblib
* Git & GitHub
* Render
* Streamlit Community Cloud

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/supreet122/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

#### Windows

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the FastAPI backend

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Start the Streamlit dashboard

Open another terminal and run:

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser.

## 🔮 Example Prediction Flow

```text
Customer Information
        ↓
Streamlit Dashboard
        ↓
FastAPI /predict
        ↓
Preprocessing
        ↓
Machine Learning Model
        ↓
Churn Prediction
        ↓
Churn Probability
        ↓
SHAP Explanation
        ↓
Dashboard Result
```

## 📊 Model Evaluation

The project includes visual evaluation reports such as:

* Confusion Matrix
* ROC Curve
* SHAP Feature Importance
* Local SHAP Feature Importance

These reports are available in the `reports/` directory.

## 🔐 Notes

The model predictions are based on the training data and features used during development. Model performance metrics reported above are based on the project's test evaluation.

## 👨‍💻 Author

**Supreet Singh**

GitHub:
https://github.com/supreet122

---

⭐ If you find this project useful, consider giving the repository a star.
