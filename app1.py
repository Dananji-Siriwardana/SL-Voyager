import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from time import sleep
import joblib
import os

# Custom CSS for ocean/beach vibe
st.markdown("""
    <style>
    .main {
        background: black;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton>button {
        background: linear-gradient(to right, #00c4b4, #1e90ff);
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
        border: 1px solid #006994;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #00e5d2, #00b7eb);
        transform: scale(1.05);
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .stSelectbox, .stNumberInput {
        background-color: black;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #00c4b4;
    }
    h1, h2, h3 {
        color: #006994;
        font-family: 'Arial', sans-serif;
        text-shadow: 1px 1px 2px rgba(0,105,148,0.3);
    }
    .stDataFrame {
        border: 2px solid #ff6f61;
        border-radius: 8px;
    }
    .metric-card {
        background-color: #e6f3fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #00c4b4;
        color: #004aad;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("forecaster-23f9d-firebase-adminsdk-fbsvc-3e068ad9d4.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Load models and encoder
try:
    encoder = joblib.load("models/season_encoder.pkl")
    models = {
        "Tourists": joblib.load("models/tourists_model.pkl"),
        "Spending": joblib.load("models/spending_model.pkl"),
        "Flight Arrivals": joblib.load("models/flight_arrivals_model.pkl"),
        "Hotel Occupancy": joblib.load("models/hotel_occupancy_model.pkl")
    }
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py to generate models.")
    st.stop()

# Title
st.title("🌊 Tourism Spending Forecaster")

# Sidebar for Historical Data Input
st.sidebar.header("Enter Historical Tourism Data")
season = st.sidebar.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
year = st.sidebar.number_input("Year", 2000, 2025, 2023)
hotel_occupancy = st.sidebar.number_input("Hotel Occupancy (%)", 0, 100, 75)
flight_arrivals = st.sidebar.number_input("Flight Arrivals", 0, 100000, 5000)
avg_spending = st.sidebar.number_input("Avg Spending ($)", 0, 10000, 500)
historical_tourists = st.sidebar.number_input("Historical Tourists", 0, 1000000, 10000)

# Save Historical Data to Firebase
if st.sidebar.button("Save Historical Data"):
    db.collection("historical_tourism").add({
        "season": season,
        "year": year,
        "hotel_occupancy": hotel_occupancy,
        "flight_arrivals": flight_arrivals,
        "avg_spending": avg_spending,
        "historical_tourists": historical_tourists
    })
    st.sidebar.success("Historical Data Saved Successfully! Please retrain models using train_model.py.")

# Fetch historical data
def fetch_historical_data():
    historical_data = db.collection("historical_tourism").stream()
    data_list = []
    doc_ids = []
    for doc in historical_data:
        data = doc.to_dict()
        data_list.append(data)
        doc_ids.append(doc.id)
    return pd.DataFrame(data_list), doc_ids

df, doc_ids = fetch_historical_data()

# Tabbed Interface
tab1, tab2, tab3 = st.tabs(["📊 Historical Data", "🔮 Predictions", "📈 Trends"])

# Tab 1: Historical Data
with tab1:
    st.subheader("Historical Tourism Data")
    if not df.empty:
        df["Document ID"] = doc_ids
        st.dataframe(df, use_container_width=True)

        # Visualization: Scatter Plot
        fig = px.scatter(df, x='flight_arrivals', y='historical_tourists', color='season', size='avg_spending',
                         title="Flight Arrivals vs Historical Tourists (Size = Avg Spending)",
                         hover_data=['year', 'hotel_occupancy'])
        fig.update_layout(
            template="plotly",
            plot_bgcolor="black",
            paper_bgcolor="black",
            font_color="#006994",
            title_font_color="#006994",
            legend_bgcolor="black"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export Data
        csv = df.to_csv(index=False)
        st.download_button("Download Historical Data as CSV", csv, "historical_tourism.csv", "text/csv")
    else:
        st.warning("⚠️ No historical data available. Add some data first.")

# Tab 2: Predictions
with tab2:
    st.subheader("Predict Future Tourism Insights")
    if not df.empty and len(df) >= 3:
        # Calculate accuracy metrics
        season_encoded = encoder.transform(df[['season']])
        season_df = pd.DataFrame(season_encoded, columns=encoder.get_feature_names_out(['season']))
        X = pd.concat([season_df, df[['year']].reset_index(drop=True)], axis=1)
        y_tourists = df['historical_tourists']
        y_spending = df['avg_spending']
        y_flights = df['flight_arrivals']
        y_occupancy = df['hotel_occupancy']

        metrics = {}
        for name, model in models.items():
            y_pred = model.predict(X)
            y_true = y_tourists if name == "Tourists" else y_spending if name == "Spending" else y_flights if name == "Flight Arrivals" else y_occupancy
            metrics[name] = {
                "R2": r2_score(y_true, y_pred),
                "MAE": mean_absolute_error(y_true, y_pred)
            }

        # User inputs for prediction
        pred_season = st.selectbox("Select Season for Prediction", ["Spring", "Summer", "Autumn", "Winter"])
        pred_year = st.number_input("Prediction Year", 2025, 2030, 2026)

        # Historical averages
        season_data = df[df['season'] == pred_season]
        if not season_data.empty:
            st.markdown(f"**Historical Averages for {pred_season}:**")
            col1, col2 = st.columns(2)
            col1.markdown(f"<div class='metric-card'>Avg Tourists: {int(season_data['historical_tourists'].mean())}</div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric-card'>Avg Spending: ${int(season_data['avg_spending'].mean())}</div>", unsafe_allow_html=True)

        # Make prediction
        if st.button("Predict Tourism Insights"):
            with st.spinner("Analyzing data..."):
                sleep(1)
                pred_season_encoded = encoder.transform([[pred_season]])
                future_data = np.hstack([pred_season_encoded, [[pred_year]]])

                # Predictions
                predictions = {name: model.predict(future_data)[0] for name, model in models.items()}

                # Confidence intervals
                confidence_intervals = {}
                for name, model in models.items():
                    y_pred = model.predict(X)
                    y_true = y_tourists if name == "Tourists" else y_spending if name == "Spending" else y_flights if name == "Flight Arrivals" else y_occupancy
                    mse = np.mean((y_true - y_pred) ** 2)
                    std_err = np.sqrt(mse / len(X))
                    confidence_intervals[name] = (predictions[name] - 1.96 * std_err, predictions[name] + 1.96 * std_err)

                # Display predictions
                st.success(f"Predicted Insights for {pred_season} {pred_year}:")
                col1, col2 = st.columns(2)
                col1.markdown(f"<div class='metric-card'>Predicted Tourists: {int(predictions['Tourists'])}<br>95% CI: [{int(confidence_intervals['Tourists'][0])}, {int(confidence_intervals['Tourists'][1])}]</div>", unsafe_allow_html=True)
                col2.markdown(f"<div class='metric-card'>Predicted Avg Spending: ${int(predictions['Spending'])}<br>95% CI: [${int(confidence_intervals['Spending'][0])}, ${int(confidence_intervals['Spending'][1])}]</div>", unsafe_allow_html=True)
                col1.markdown(f"<div class='metric-card'>Predicted Flight Arrivals: {int(predictions['Flight Arrivals'])}<br>95% CI: [{int(confidence_intervals['Flight Arrivals'][0])}, {int(confidence_intervals['Flight Arrivals'][1])}]</div>", unsafe_allow_html=True)
                col2.markdown(f"<div class='metric-card'>Predicted Hotel Occupancy: {int(predictions['Hotel Occupancy'])}%<br>95% CI: [{int(confidence_intervals['Hotel Occupancy'][0])}, {int(confidence_intervals['Hotel Occupancy'][1])}]</div>", unsafe_allow_html=True)

                # Trend analysis
                trend_tourists = "Increasing" if predictions['Tourists'] > season_data['historical_tourists'].mean() else "Decreasing"
                trend_spending = "Increasing" if predictions['Spending'] > season_data['avg_spending'].mean() else "Decreasing"
                st.markdown(f"**Trends**: Tourists: {trend_tourists} | Spending: {trend_spending}")

                # Accuracy metrics
                st.subheader("Model Accuracy")
                for name, metric in metrics.items():
                    st.markdown(f"**{name} Model**: R² = {metric['R2']:.2f}, MAE = {metric['MAE']:.2f}")

                # Export predictions
                pred_df = pd.DataFrame({
                    "Season": [pred_season],
                    "Year": [pred_year],
                    "Predicted Tourists": [int(predictions['Tourists'])],
                    "Predicted Spending": [int(predictions['Spending'])],
                    "Predicted Flight Arrivals": [int(predictions['Flight Arrivals'])],
                    "Predicted Hotel Occupancy": [int(predictions['Hotel Occupancy'])]
                })
                csv = pred_df.to_csv(index=False)
                st.download_button("Download Prediction Report", csv, "prediction_report.csv", "text/csv")

                # Model insights
                st.subheader("Model Insights")
                for name, model in models.items():
                    st.write(f"**{name} Feature Importance (Coefficients)**:")
                    for feature, coef in zip(X.columns, model.coef_):
                        st.write(f"{feature}: {coef:.2f}")
    else:
        st.warning("⚠️ Not enough historical data for prediction. Add at least 3 records.")

# Tab 3: Trends
with tab3:
    st.subheader("Seasonal Tourism Trends")
    if not df.empty:
        # Season filter
        trend_season = st.selectbox("Select Season to View Trends", ["All"] + list(df['season'].unique()))

        # Create trend plots
        fig_tourists = go.Figure()
        fig_spending = go.Figure()
        
        seasons = df['season'].unique() if trend_season == "All" else [trend_season]
        
        for s in seasons:
            season_trend = df[df['season'] == s].sort_values('year')
            if len(season_trend) < 2:
                st.warning(f"⚠️ Not enough data for {s} to display trends.")
                continue

            # Historical tourists
            fig_tourists.add_trace(go.Scatter(
                x=season_trend['year'], 
                y=season_trend['historical_tourists'],
                mode='lines+markers', 
                name=f"{s} Tourists",
                line=dict(color="#00c4b4"),
                marker=dict(size=8)
            ))

            # Historical spending
            fig_spending.add_trace(go.Scatter(
                x=season_trend['year'], 
                y=season_trend['avg_spending'],
                mode='lines+markers', 
                name=f"{s} Spending",
                line=dict(color="#ff6f61"),
                marker=dict(size=8)
            ))

            # Add annotations for peak points
            max_tourist_year = season_trend.loc[season_trend['historical_tourists'].idxmax()]
            fig_tourists.add_annotation(
                x=max_tourist_year['year'], 
                y=max_tourist_year['historical_tourists'],
                text=f"Peak: {int(max_tourist_year['historical_tourists'])}",
                showarrow=True, 
                arrowhead=2, 
                ax=20, 
                ay=-30,
                font=dict(color="#006994")
            )

            max_spending_year = season_trend.loc[season_trend['avg_spending'].idxmax()]
            fig_spending.add_annotation(
                x=max_spending_year['year'], 
                y=max_spending_year['avg_spending'],
                text=f"Peak: ${int(max_spending_year['avg_spending'])}",
                showarrow=True, 
                arrowhead=2, 
                ax=20, 
                ay=-30,
                font=dict(color="#006994")
            )

            # Predict future values (2026–2030)
            future_years = list(range(2026, 2031))
            pred_season_encoded = encoder.transform([[s]])
            future_data = np.hstack([np.repeat(pred_season_encoded, len(future_years), axis=0), [[y] for y in future_years]])
            
            future_tourists = models["Tourists"].predict(future_data)
            future_spending = models["Spending"].predict(future_data)

            # Add future predictions
            fig_tourists.add_trace(go.Scatter(
                x=future_years, 
                y=future_tourists,
                mode='lines+markers', 
                name=f"{s} Predicted Tourists",
                line=dict(dash='dash', color="#00e5d2"),
                marker=dict(size=8)
            ))

            fig_spending.add_trace(go.Scatter(
                x=future_years, 
                y=future_spending,
                mode='lines+markers', 
                name=f"{s} Predicted Spending",
                line=dict(dash='dash', color="#ff9a8b"),
                marker=dict(size=8)
            ))

        # Update layouts
        fig_tourists.update_layout(
            title=f"Tourist Trends {'by Season' if trend_season == 'All' else f'for {trend_season}'}",
            xaxis=dict(title="Year", color="#006994", gridcolor="#e6f3fa"),
            yaxis=dict(title="Tourists", color="#006994", gridcolor="#e6f3fa"),
            plot_bgcolor="black",
            paper_bgcolor="black",
            font_color="#006994",
            title_font_color="#006994",
            legend_bgcolor="black",
            showlegend=True
        )

        fig_spending.update_layout(
            title=f"Spending Trends {'by Season' if trend_season == 'All' else f'for {trend_season}'}",
            xaxis=dict(title="Year", color="#006994", gridcolor="#e6f3fa"),
            yaxis=dict(title="Avg Spending ($)", color="#006994", gridcolor="#e6f3fa"),
            plot_bgcolor="black",
            paper_bgcolor="black",
            font_color="#006994",
            title_font_color="#006994",
            legend_bgcolor="black",
            showlegend=True
        )

        # Display plots
        st.plotly_chart(fig_tourists, use_container_width=True)
        st.plotly_chart(fig_spending, use_container_width=True)

        # Export trend data
        trend_data = df[['season', 'year', 'historical_tourists', 'avg_spending']].copy()
        future_trend_data = []
        for s in seasons:
            future_years = list(range(2026, 2031))
            pred_season_encoded = encoder.transform([[s]])
            future_data = np.hstack([np.repeat(pred_season_encoded, len(future_years), axis=0), [[y] for y in future_years]])
            future_tourists = models["Tourists"].predict(future_data)
            future_spending = models["Spending"].predict(future_data)
            for y, t, sp in zip(future_years, future_tourists, future_spending):
                future_trend_data.append({"season": s, "year": y, "historical_tourists": t, "avg_spending": sp})
        
        trend_df = pd.concat([trend_data, pd.DataFrame(future_trend_data)], ignore_index=True)
        csv = trend_df.to_csv(index=False)
        st.download_button("Download Trend Data as CSV", csv, "trend_data.csv", "text/csv")
    else:
        st.warning("⚠️ No data available for trends.")

# Delete Historical Data
st.subheader("Manage Data")
if doc_ids:
    delete_id = st.selectbox("Select Historical Data to Delete", doc_ids)
    if st.button("Delete Historical Data"):
        db.collection("historical_tourism").document(delete_id).delete()
        st.success("Historical Data Deleted Successfully! Please retrain models using train_model.py.")
else:
    st.warning("⚠️ No historical data available for deletion.")