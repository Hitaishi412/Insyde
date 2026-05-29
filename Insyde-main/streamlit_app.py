import streamlit as st
import pandas as pd

st.set_page_config(page_title="Insyde UEBA", layout="wide")

st.title("Insyde UEBA Dashboard")
st.subheader("Insider Threat Detection System")

try:
    df = pd.read_csv("cert_risk_scored.csv")

    st.success("Data Loaded Successfully!")

    st.dataframe(df.head())

    if "risk_score" in df.columns:
        st.subheader("High Risk Users")

        high_risk = df[df["risk_score"] > 70]

        st.dataframe(high_risk)

except Exception as e:
    st.error(f"Error loading file: {e}")