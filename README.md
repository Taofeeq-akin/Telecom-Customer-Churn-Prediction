Telecom Customer Churn Prediction/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── validation.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation_thresholds.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── threshold_optimization.py
│
├── models/
│   ├── baseline_logistic.pkl
│   ├── xgboost_fraud.pkl
│   └── threshold.json
│
├── app/
│   ├── app.py                # Streamlit app
│   ├── utils.py              # Model loading & prediction
│
├── reports/
│   ├── confusion_matrices/
│   ├── pr_curves/
│   └── business_summary.md
│
├── requirements.txt
├── README.md
└── .gitignore