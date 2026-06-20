import os
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
)

from reportlab.lib.styles import getSampleStyleSheet


def generate_pdf_report(
    results,
    risk_chart_path,
    top_users_chart_path,
    output_path="Insider_Threat_Report.pdf",
):
    pdf = SimpleDocTemplate(output_path)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Insider Threat Detection Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Generated On: {generated_time}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Statistics calculation
    total_users = len(results)
    high_risk = len(results[results["risk_level"] == "High"]) if total_users > 0 else 0
    medium_risk = len(results[results["risk_level"] == "Medium"]) if total_users > 0 else 0
    low_risk = len(results[results["risk_level"] == "Low"]) if total_users > 0 else 0

    highest_user = None
    if total_users > 0 and "risk_score" in results.columns:
        highest_user = results.loc[results["risk_score"].idxmax()]

    elements.append(Paragraph("Executive Summary", styles["Heading1"]))

    if highest_user is not None:
        summary = (
            f"Total Users Analyzed: {total_users}<br/>"
            f"High Risk Users: {high_risk}<br/>"
            f"Medium Risk Users: {medium_risk}<br/>"
            f"Low Risk Users: {low_risk}<br/><br/>"
            f"Highest Risk User: {highest_user['user']}<br/>"
            f"Risk Score: {highest_user['risk_score']:.2f}<br/>"
            f"Risk Level: {highest_user['risk_level']}<br/><br/>"
            f"Threat Indicators:<br/>{highest_user['reason']}"
        )
    else:
        summary = (
            f"Total Users Analyzed: {total_users}<br/>"
            f"High Risk Users: {high_risk}<br/>"
            f"Medium Risk Users: {medium_risk}<br/>"
            f"Low Risk Users: {low_risk}<br/><br/>"
            f"Highest Risk User: N/A<br/>"
        )

    elements.append(Paragraph(summary, styles["BodyText"]))
    elements.append(Spacer(1, 20))

    # --- Helper Path-Resolution Logic ---
    # This checks if the chart exists where Streamlit claims it is. 
    # If not, it drops any redundant "Dynamic/" prefix caused by working-directory shifts.
    def resolve_chart_path(path):
        if not path:
            return None
        if os.path.exists(path):
            return path
        
        # Alternative attempt: strip 'Dynamic/' if we are already inside it
        alt_path = path.replace("Dynamic/", "") if "Dynamic/" in path else f"Dynamic/{path}"
        if os.path.exists(alt_path):
            return alt_path
            
        # Hard fallback: Check standard reporting directory inside current location
        basename = os.path.basename(path)
        fallback_path = os.path.join("reporting", basename)
        if os.path.exists(fallback_path):
            return fallback_path
            
        return None

    resolved_risk_path = resolve_chart_path(risk_chart_path)
    resolved_top_path = resolve_chart_path(top_users_chart_path)

    # --- Risk Distribution Chart Rendering ---
    if resolved_risk_path:
        elements.append(Paragraph("Risk Distribution", styles["Heading1"]))
        elements.append(Image(resolved_risk_path, width=450, height=300))
        elements.append(Spacer(1, 20))
    else:
        # Debug helper added directly to PDF layout text to help you track failures
        elements.append(Paragraph("Risk Distribution Chart File Not Found", styles["Heading1"]))
        elements.append(Paragraph(f"<i>Checked paths: '{risk_chart_path}'</i>", styles["Normal"]))
        elements.append(Spacer(1, 20))

    # --- Top Users Chart Rendering ---
    if resolved_top_path:
        elements.append(Paragraph("Top Risk Users", styles["Heading1"]))
        elements.append(Image(resolved_top_path, width=450, height=300))
        elements.append(Spacer(1, 20))
    else:
        elements.append(Paragraph("Top Risk Users Chart File Not Found", styles["Heading1"]))
        elements.append(Paragraph(f"<i>Checked paths: '{top_users_chart_path}'</i>", styles["Normal"]))
        elements.append(Spacer(1, 20))

    # High Risk Users list
    elements.append(Paragraph("High Risk Users", styles["Heading1"]))
    high_risk_df = results[results["risk_level"] == "High"] if total_users > 0 else []

    if len(high_risk_df) > 0:
        for _, row in high_risk_df.iterrows():
            elements.append(
                Paragraph(
                    (
                        f"<b>User:</b> {row['user']}<br/>"
                        f"<b>Risk Score:</b> {row['risk_score']:.2f}<br/>"
                        f"<b>Risk Level:</b> {row['risk_level']}<br/>"
                        f"<b>Reason:</b> {row['reason']}"
                    ),
                    styles["BodyText"],
                )
            )
            elements.append(Spacer(1, 10))
    else:
        elements.append(Paragraph("No High Risk Users Found.", styles["BodyText"]))

    elements.append(PageBreak())

    # Recommendations
    elements.append(Paragraph("Security Recommendations", styles["Heading1"]))
    recommendations = (
        "• Investigate users with high risk scores.<br/><br/>"
        "• Review abnormal file access activity.<br/><br/>"
        "• Audit USB device usage patterns.<br/><br/>"
        "• Monitor after-hours access attempts.<br/><br/>"
        "• Review external email communication activity.<br/><br/>"
        "• Conduct periodic insider threat assessments."
    )
    elements.append(Paragraph(recommendations, styles["BodyText"]))

    # Cleaned up code sequence: Build doc first, THEN return output
    pdf.build(elements)
    return output_path
