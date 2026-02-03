import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="wide"
)

from pathlib import Path
import streamlit as st
import joblib

# ---------------------------------
# Project folders (repo-relative)
# ---------------------------------
ARTIFACTS_DIR = Path("artifacts")
BASE_DATA = Path("Data") / "raw"

# ---------------------------------
# Load assets (MODEL ONLY)
# ---------------------------------
@st.cache_resource
def load_assets():
    model_path = ARTIFACTS_DIR / "random_forest_model.pkl"
    threshold_path = ARTIFACTS_DIR / "random_forest_model_score.pkl"

    # ---- Model check ----
    if not model_path.is_file():
        st.error("❌ random_forest_model.pkl not found in artifacts/")
        st.stop()

    model = joblib.load(model_path)

    # ---- Feature schema check ----
    if not hasattr(model, "feature_names_in_"):
        st.error(
            "❌ Model does not contain feature names.\n"
            "It must be trained using a pandas DataFrame."
        )
        st.stop()

    model_features = list(model.feature_names_in_)

    # ---- Threshold (optional) ----
    if threshold_path.is_file():
        threshold = joblib.load(threshold_path)
    else:
        st.warning("⚠️ Threshold file not found. Using default threshold = 0.5")
        threshold = 0.5

    return model, threshold, model_features


@st.cache_data
def load_raw_data():
    return pd.read_csv(BASE_DATA / "WA_Fn-UseC_-Telco-Customer-Churn.csv")


df = load_raw_data()
model, threshold, model_features = load_assets()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Data Overview", "🧠 Make Prediction"]
)

# -----------------------------
# HOME
# -----------------------------
if page == "🏠 Home":
    st.title("🤖 Telco Churn Prediction App")
    st.markdown("""
    - Random Forest model  
    - Full feature engineering  
    - One-hot encoding  
    - **46-feature inference (model-verified)**  
    """)
    st.success(f"Decision Threshold: {threshold}")

# ----------------------------- # DATA OVERVIEW # ----------------------------- 
elif page == "📊 Data Overview": 
    st.title("📊 Training Dataset Overview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)

# -----------------------------
# PREDICTION
# -----------------------------
elif page == "🧠 Make Prediction":
    st.title("🧠 Make a Prediction")

    # Raw input columns
    start_col = "gender"
    input_columns = df.loc[:, start_col:].columns[:-1]

    user_input = {}
    for col in input_columns:
        if df[col].dtype == "object":
            user_input[col] = st.selectbox(col, sorted(df[col].dropna().unique()))
        else:
            user_input[col] = st.number_input(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].median())
            )

    input_df = pd.DataFrame([user_input])
    st.subheader("🟡 Raw Input")
    st.dataframe(input_df)

    if st.button("🚀 Predict"):

        # -----------------------------
        # Fix dtypes
        # -----------------------------
        input_df["TotalCharges"] = pd.to_numeric(
            input_df["TotalCharges"], errors="coerce"
        ).fillna(0)

        # -----------------------------
        # Feature Engineering
        # -----------------------------
        fe_df = input_df.copy()

        fe_df["TotalCharges_per_tenure"] = fe_df["TotalCharges"] / (fe_df["tenure"] + 1)
        fe_df["MonthlyCharges_tenure"] = fe_df["MonthlyCharges"] * fe_df["tenure"]
        fe_df["TotalCharges_per_Monthly"] = fe_df["TotalCharges"] / (fe_df["MonthlyCharges"] + 1)

        fe_df["is_new_customer"] = (fe_df["tenure"] == 1).astype(int)
        fe_df["is_month_to_month"] = (fe_df["Contract"] == "Month-to-month").astype(int)

        fe_df["short_tenure"] = (fe_df["tenure"] < 12).astype(int)
        fe_df["medium_tenure"] = ((fe_df["tenure"] >= 12) & (fe_df["tenure"] < 36)).astype(int)
        fe_df["long_tenure"] = (fe_df["tenure"] >= 36).astype(int)

        fe_df["high_monthly_charge"] = (
            fe_df["MonthlyCharges"] > df["MonthlyCharges"].median()
        ).astype(int)

        fe_df["price_pressure"] = fe_df["MonthlyCharges"] / (fe_df["TotalCharges_per_tenure"] + 1)

        fe_df["no_support_or_security"] = (
            (fe_df["TechSupport"] == "No") &
            (fe_df["OnlineSecurity"] == "No")
        ).astype(int)

        fe_df["no_protection_bundle"] = (
            (fe_df["OnlineBackup"] == "No") &
            (fe_df["DeviceProtection"] == "No") &
            (fe_df["TechSupport"] == "No")
        ).astype(int)

        fe_df["fiber_user"] = (fe_df["InternetService"] == "Fiber optic").astype(int)
        fe_df["electronic_check"] = (fe_df["PaymentMethod"] == "Electronic check").astype(int)
        fe_df["paperless_billing"] = (fe_df["PaperlessBilling"] == "Yes").astype(int)

        fe_df["new_and_expensive"] = (
            (fe_df["tenure"] <= 6) &
            (fe_df["MonthlyCharges"] > df["MonthlyCharges"].median())
        ).astype(int)

        # -----------------------------
        # Encoding + alignment (MODEL-DRIVEN)
        # -----------------------------
        fe_encoded = pd.get_dummies(fe_df)
        fe_encoded = fe_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)

        fe_encoded = fe_encoded.reindex(
            columns=model_features,
            fill_value=0
        )

        st.subheader("🔵 Final Model Input (46 Columns)")
        st.dataframe(fe_encoded)

        # -----------------------------
        # Predict
        # -----------------------------
        X = fe_encoded.values
        probability = model.predict_proba(X)[0][1]
        prediction = int(probability >= threshold)

        st.subheader("🔍 Prediction Result")
        st.write(f"Probability: {probability:.2%}")

        if prediction == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is likely to STAY")
