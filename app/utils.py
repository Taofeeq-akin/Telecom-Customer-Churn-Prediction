import streamlit as st
import pandas as pd
import numpy as np
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