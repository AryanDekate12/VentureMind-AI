import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="VentureMind AI", layout="wide")

st.title("🚀 VentureMind AI")
st.subheader("Startup Intelligence Platform")

st.write("AI-powered platform that predicts startup success and analyzes startup potential.")

st.divider()

# Sidebar Inputs
st.sidebar.header("Startup Parameters")

funding = st.sidebar.slider("Funding (Million $)", 0, 200, 50)
team_size = st.sidebar.slider("Team Size", 1, 100, 10)
market_size = st.sidebar.slider("Market Size (Billion $)", 1, 500, 100)
years = st.sidebar.slider("Years Operating", 0, 10, 2)
experience = st.sidebar.slider("Founder Experience (Years)", 0, 20, 5)

features = np.array([[funding, team_size, market_size, years, experience]])

# Model prediction
prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0][1]

score = int(probability * 100)

# Risk Level
if probability > 0.75:
    risk = "Low Risk"
elif probability > 0.45:
    risk = "Medium Risk"
else:
    risk = "High Risk"

st.divider()

# Main Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Startup Score", f"{score}/100")
col2.metric("Success Probability", f"{round(probability*100,2)}%")
col3.metric("Risk Level", risk)

st.divider()

# Prediction Result
st.subheader("Prediction Result")

if prediction == 1:
    st.success("This startup has strong success potential 🚀")
else:
    st.error("This startup has high failure risk ⚠️")

st.divider()

# Investor Recommendation
st.subheader("Investor Recommendation")

if funding < 20:
    investor = "Angel Investors"
elif funding < 100:
    investor = "Seed / Early Venture Capital"
else:
    investor = "Large Venture Capital Firms"

st.write(f"Recommended Investor Type: **{investor}**")

st.divider()

# Startup Valuation Estimate
st.subheader("Startup Valuation Estimate")

valuation = (
    funding * 4 +
    team_size * 2 +
    market_size * 1.5 +
    experience * 3
)

st.metric("Estimated Startup Valuation", f"${int(valuation)}M")

st.divider()

# Market Opportunity Score
st.subheader("Market Opportunity Analysis")

market_score = int((market_size / 500) * 100)

st.metric("Market Opportunity Score", f"{market_score}/100")

if market_score > 70:
    st.success("Huge Market Opportunity 🌍")
elif market_score > 40:
    st.warning("Moderate Market Opportunity")
else:
    st.error("Small Market Opportunity")

st.divider()

# Founder Strength
st.subheader("Founder Strength Analysis")

founder_score = int((experience / 20) * 100)

st.metric("Founder Strength Score", f"{founder_score}/100")

if founder_score > 70:
    st.success("Highly Experienced Founder")
elif founder_score > 40:
    st.warning("Moderately Experienced Founder")
else:
    st.error("Inexperienced Founder")

st.divider()

# Feature Importance Chart
st.subheader("Feature Importance Analysis")

importance = model.feature_importances_

feature_names = [
    "Funding",
    "Team Size",
    "Market Size",
    "Years Operating",
    "Founder Experience"
]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

fig, ax = plt.subplots()

ax.barh(importance_df["Feature"], importance_df["Importance"])
ax.set_xlabel("Importance Score")
ax.set_title("Factors Influencing Startup Success")

st.pyplot(fig)

st.divider()

# Radar Chart
st.subheader("Startup Profile Radar")

radar_data = pd.DataFrame({
    "Metric": ["Funding", "Team", "Market", "Years", "Experience"],
    "Value": [funding, team_size, market_size, years, experience]
})

fig2 = px.line_polar(
    radar_data,
    r="Value",
    theta="Metric",
    line_close=True
)

st.plotly_chart(fig2)

st.divider()

# Startup Health Gauge
st.subheader("Startup Health Score")

fig3 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score,
    title={'text': "Startup Health"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "green"},
        'steps': [
            {'range': [0, 40], 'color': "red"},
            {'range': [40, 70], 'color': "yellow"},
            {'range': [70, 100], 'color': "lightgreen"}
        ]
    }
))

st.plotly_chart(fig3)

st.divider()

# Startup Report
st.subheader("Startup Analysis Report")

report = f"""
Startup Score: {score}/100

Success Probability: {round(probability*100,2)}%

Risk Level: {risk}

Recommended Investor: {investor}

Estimated Valuation: ${int(valuation)}M

Market Opportunity Score: {market_score}/100

Founder Strength Score: {founder_score}/100
"""

st.download_button(
    label="Download Startup Report",
    data=report,
    file_name="startup_report.txt"
)

st.divider()

st.caption("VentureMind AI — Startup Intelligence Platform")