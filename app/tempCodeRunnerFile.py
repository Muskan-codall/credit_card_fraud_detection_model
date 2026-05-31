from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "fraud_detection_rf.pkl"
import streamlit as st

st.write("ROOT_DIR:", ROOT_DIR)
st.write("MODEL_PATH:", MODEL_PATH)
st.write("Exists:", MODEL_PATH.exists())