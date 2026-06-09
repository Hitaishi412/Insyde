import os
import sys
from evaluation import evaluate_model
import pandas as pd
import streamlit as st
import plotly.express as px
from prediction.predict import analyze_threats
from reporting.generate_pdf import generate_pdf_report

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from feature_pipeline import process_logs
from prediction.predict import analyze_threats

st.set_page_config(
    page_title="Insider Threat Detection",
    layout="wide"
)

st.title("🔐 Insider Threat Detection System")

st.markdown(
    """
    Upload any CSV dataset.

    Supported formats:
    - Email Logs
    - Logon Logs
    - File Activity Logs
    - Device Logs
    - Combined Activity Logs
    - Pre-engineered Feature Dataset
    """
)

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        raw_df = pd.read_csv(uploaded_file)

        y_true = None

        if "label" in raw_df.columns:
            y_true = raw_df["label"]
        st.subheader("Uploaded Dataset")
        st.dataframe(raw_df.head())

        st.write(f"Rows: {raw_df.shape[0]}")
        st.write(f"Columns: {raw_df.shape[1]}")

        # --------------------------
        # Feature Engineering
        # --------------------------

        feature_df = process_logs(raw_df)
        st.write("Feature Columns")
        st.write(feature_df.columns.tolist())
        st.subheader("Generated Behavioral Features")
        st.dataframe(feature_df)

        # --------------------------
        # Threat Analysis
        # --------------------------

        results = analyze_threats(feature_df)

        # Model Evaluation
        st.write("Labels Found:", y_true is not None)

        if y_true is not None:
            st.write("Label Distribution")
            st.write(y_true.value_counts())
        if y_true is not None:

            metrics = evaluate_model(
            y_true,
            results["anomaly"]
        )

            st.subheader("📈 Model Performance")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(
                    "Accuracy",
                    f"{metrics['accuracy']:.2%}"
                )

            with col2:
                st.metric(
                    "Precision",
                    f"{metrics['precision']:.2%}"
                )

            with col3:
                st.metric(
                    "Recall",
                    f"{metrics['recall']:.2%}"
                )

            with col4:
                st.metric(
                    "F1 Score",
                    f"{metrics['f1_score']:.2%}"
                )

            with col5:
                st.metric(
                    "ROC AUC",
                    f"{metrics['roc_auc']:.2%}"
                )

            st.write("Confusion Matrix")

            st.write(
                metrics["confusion_matrix"]
            )

        st.subheader("Threat Analysis Results")

        display_results = (
            results[
                [
                    "user",
                    "anomaly_score",
                    "risk_score",
                    "risk_level",
                    "reason"
                ]
            ]
            .sort_values(
                "risk_score",
                ascending=False
            )
        )

        st.dataframe(
            display_results.style.background_gradient(
                subset=["risk_score"],
                cmap="Reds"
            )
        )

        # --------------------------
        # Metrics
        # --------------------------

        total_users = len(results)

        high_risk = results[
            results["risk_level"] == "High"
        ]

        medium_risk = results[
            results["risk_level"] == "Medium"
        ]

        low_risk = results[
            results["risk_level"] == "Low"
        ]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Users",
                total_users
            )

        with col2:
            st.metric(
                "High Risk",
                len(high_risk)
            )

        with col3:
            st.metric(
                "Medium Risk",
                len(medium_risk)
            )

        with col4:
            st.metric(
                "Low Risk",
                len(low_risk)
            )

        # --------------------------
        # Executive Summary
        # --------------------------

        highest_user = results.loc[
            results["risk_score"].idxmax()
        ]

        st.subheader("📋 Executive Summary")

        st.info(
            f"""
            Highest Risk User: {highest_user['user']}

            Risk Score: {highest_user['risk_score']:.2f}

            Risk Level: {highest_user['risk_level']}
            """
        )

        st.write("Primary Threat Indicators:")
        st.write(highest_user["reason"])

        # --------------------------
        # Risk Distribution Pie Chart
        # --------------------------

        st.subheader("📊 Risk Distribution")

        risk_counts = (
            results["risk_level"]
            .value_counts()
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Level",
            "Count"
        ]

        risk_fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Count",
            title="Risk Level Distribution",
            color="Risk Level",
            color_discrete_map={
                "Low": "green",
                "Medium": "orange",
                "High": "red"
            }
        )

        risk_fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        st.plotly_chart(
            risk_fig,
            use_container_width=True
        )

        risk_fig.write_image(
            "risk_distribution.png",
            width=1200,
            height=800,
            scale=2
        )
        # --------------------------
        # Top 10 Risk Users
        # --------------------------

        st.subheader(
            "🚨 Top 10 Highest Risk Users"
        )

        top_users = (
            results
            .sort_values(
                "risk_score",
                ascending=False
            )
            .head(10)
        )

        top_users_fig = px.bar(
            top_users,
            x="user",
            y="risk_score",
            color="risk_level",
            title="Top Risk Users",
            color_discrete_map={
                "Low": "green",
                "Medium": "orange",
                "High": "red"
            }
        )

        top_users_fig.update_layout(
            xaxis_title="Employee",
            yaxis_title="Risk Score"
        )

        st.plotly_chart(
            top_users_fig,
            use_container_width=True
        )

        top_users_fig.write_image(
            "top_users.png",
            width=1200,
            height=800,
            scale=2
        )

        # --------------------------
        # Anomaly Distribution
        # --------------------------

        st.subheader(
            "📈 Anomaly Score Distribution"
        )

        anomaly_fig = px.histogram(
            results,
            x="anomaly_score",
            nbins=20,
            title="Anomaly Score Distribution",
            color_discrete_sequence=["crimson"]
        )

        st.plotly_chart(
            anomaly_fig,
            use_container_width=True
        )
        # --------------------------
        # High Risk Users
        # --------------------------

        st.subheader(
            "🚨 High Risk Users"
        )

        if len(high_risk) > 0:

            st.dataframe(
                high_risk[
                    [
                        "user",
                        "anomaly_score",
                        "risk_score",
                        "risk_level",
                        "reason"
                    ]
                ]
                .sort_values(
                    "risk_score",
                    ascending=False
                )
            )

        else:

            st.success(
                "No High Risk Users Found"
            )

        # --------------------------
        # Recommendations
        # --------------------------

        st.subheader(
            "🛡 Security Recommendations"
        )

        if len(high_risk) > 0:

            st.warning(
                """
                • Investigate high-risk users immediately

                • Review abnormal file access activity

                • Audit USB device usage

                • Monitor after-hours access

                • Review external email communications
                """
            )

        else:

            st.success(
                """
                No critical insider threat indicators detected.

                Continue routine monitoring.
                """
            )

        # --------------------------
        # Download CSV Report
        # --------------------------

        csv = results.to_csv(
            index=False
        )

        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name="threat_report.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

    # --------------------------
    # Generate PDF Report
    # --------------------------

    pdf_path = generate_pdf_report(
        results=results,
        risk_chart_path="risk_distribution.png",
        top_users_chart_path="top_users.png"
    )

    with open(pdf_path, "rb") as pdf_file:

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_file,
            file_name="Insider_Threat_Report.pdf",
            mime="application/pdf"
        )