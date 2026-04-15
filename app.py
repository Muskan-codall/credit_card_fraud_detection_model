import streamlit as st
import pandas as pd
import numpy as np
import matplotlip.pyplot as plt
import seaborn as sns
import pickle
import pyplot.graph_object as go
import plotly.express as px
from pathlib import path
from plotly.subplots import make_subplots

st.set_page_config(
   page_title="Fraud credit card detection"
   page_icon="📊"
   layout="wide"
   initial_sidebar_state="expanded"
)

st.markdown(body="""
    <style>
    .main_header{
            font-size : 2.5rem;
            font-weight : bold;
            colour : #1233885
            text-align : centre;
            margin-bottom : 2rem;
            }
               
    .prediction-card{
            padding : 1.5rem;
            border :10px;
            margin:1rem 0;
            box-shadow:0 2px 4px rgba(0,0,0,0.1);
            }
    .metric-card{
            background-clr : #f8f9fa;
            padding: 1rem;
            border-radius:8px;
            text-align: centre;
        }
        </style>
     """
unsafe_allow_html=True)

@st.cache_resource
def load_model_artifact():
    """Load trained model and artifacts"""

    try:
        model_path= path(__file__).parent / "models"/ "credit_card_model.pkl"
        columns_path= path(__file__).parent / "models"/ "credit_card_model.pkl"
   

        if not model_path.exist():
           model_path = path(__file__).parent.parent /"models"/"credit_card_model.pkl"
           columns_path = path(__file__).parent.parent /"models"/"credit_card_model.pkl"
   
        model = pickle.load(open("model_path","rb"))
        model_colums = pickle.load(open("model_path","rb"))
        return model, model_colums

    except Exception as e:
            st.error(f"error loading model:{e}")
            return None ,None
    
@st.cache_data
def load_real_data():
     df= pd.read_csv("../data/raw//creditcard.csv")
     
