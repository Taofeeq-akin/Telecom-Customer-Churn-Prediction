# 📌 Telecom Customer Churn Prediction
### Telco Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Telcom ML](https://img.shields.io/badge/Domain-Telcom-red)
![Status](https://img.shields.io/badge/Status-Deployed-success)

---

Predicting customer churn helps telecom companies identify customers likely to leave so they can take proactive retention actions.

This project applies machine learning techniques to classify customers as **Churn** or **Non-Churn** using historical telecom customer data.

---

## 🔍 Dataset

The dataset contains customer demographic, account, and service usage information.

- **Target Variable:** `Churn` (Yes/No)
- **Key Features:**
  - Tenure
  - MonthlyCharges
  - TotalCharges
  - Contract
  - PaymentMethod
  - InternetService
  - OnlineSecurity
  - TechSupport

---

## 🛠️ Project Workflow

### 1️⃣ Exploratory Data Analysis (EDA)
- Analyzed churn distribution
- Identified key patterns and relationships
- Visualized feature correlations

### 2️⃣ Data Preprocessing
- Handled missing values
- Encoded categorical variables
- Performed feature engineering
- Split dataset into training and testing sets

### 3️⃣ Model Building
- Trained multiple classification models
- Selected **Random Forest Classifier** as final model

### 4️⃣ Model Evaluation
Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📈 Model Performance

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.76  |
| Precision | 0.53  |
| Recall    | 0.81  |
| F1-Score  | 0.64  |


---

## 📊 Key Insights

- Customers with shorter tenure showed higher churn rates.
- Contract type significantly influenced churn behavior.
- Monthly charges were positively correlated with churn likelihood.
- Random Forest handled feature interactions effectively and reduced overfitting.

---

## 🚀 Tech Stack

- **Language:** Python
- **Libraries:** 
  - Data Processing: Pandas, NumPy
  - Machine Learning: Scikit-learn
  - Visualization: Matplotlib / Seaborn
- **Environment:** Jupyter Notebook

---

### 🔮 Future Improvements

Hyperparameter tuning using GridSearchCV

Testing advanced ensemble models (XGBoost, LightGBM)

Handling class imbalance more robustly

Deploying model with Streamlit or Flask

### 👩‍⚕️ About the Author

Healthcare Data Scientist & Machine Learning Engineer

I build end-to-end machine learning systems; from data cleaning and modeling to evaluation and deployment, with a strong focus on accuracy, interpretability, and impact.

📬 Open to:

Internship / Entry-level / junior ML roles

Aspiring Machine Learning Engineer

LinkedIn: https://linkedin.com/in/taofeeq-akintunde
