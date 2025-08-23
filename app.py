import streamlit as st
import pandas as pd
import joblib
import random
from datetime import datetime

# -------------------------------
# Load trained model, scaler & encoders
# -------------------------------
model = joblib.load("training_results/fraud_detection_model.pkl")
scaler = joblib.load("training_results/scaler.pkl")
encoders = joblib.load("training_results/encoders.pkl")
feature_columns = joblib.load("training_results/feature_columns.pkl")

# -------------------------------
# Load smaller pre-saved dataset (balanced fraud/not fraud)
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("fraudSample.csv")   # <-- pre-created small file

full_data = load_data()

# -------------------------------
# Build category mapping dynamically
# -------------------------------
raw_categories = sorted(full_data["category"].unique())
category_mapping = {cat: f"{cat}" for cat in raw_categories}
inverse_category_mapping = {v: k for k, v in category_mapping.items()}

# Job dropdown options
job_options = sorted(list(encoders["job"].classes_))

# -------------------------------
# Helper to pick random row
# -------------------------------
def pick_sample(fraud_type="random"):
    if fraud_type == "fraud":
        row = full_data[full_data["is_fraud"] == 1].sample(1, random_state=random.randint(0, 10000)).iloc[0]
    elif fraud_type == "notfraud":
        row = full_data[full_data["is_fraud"] == 0].sample(1, random_state=random.randint(0, 10000)).iloc[0]
    else:
        row = full_data.sample(1, random_state=random.randint(0, 10000)).iloc[0]

    return {
        "trans_date": pd.to_datetime(row["trans_date_trans_time"]).date(),
        "trans_time": pd.to_datetime(row["trans_date_trans_time"]).strftime("%H:%M"),
        "category": category_mapping.get(row["category"], row["category"]),
        "amt": row["amt"],
        "gender": "Male" if row["gender"] == "M" else "Female",
        "job": row["job"],
        "lat": row["lat"],
        "long": row["long"],
        "city_pop": row["city_pop"],
        "merch_lat": row["merch_lat"],
        "merch_long": row["merch_long"]
    }

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Fraud Detection System", page_icon="💳", layout="wide")
st.title("💳 Fraud Detection System")
st.write("Enter transaction details manually or autofill from dataset:")

# Autofill buttons
col1, col2, col3 = st.columns(3)

if col1.button("⚠️ Autofill Fraud Case"):
    st.session_state.autofill = pick_sample("fraud")

if col2.button("✅ Autofill Not Fraud Case"):
    st.session_state.autofill = pick_sample("notfraud")

if col3.button("🎲 Autofill Random Case"):
    st.session_state.autofill = pick_sample("random")


# Get defaults
defaults = st.session_state.get("autofill", {})

# -------------------------------
# Input Form
# -------------------------------
with st.form("fraud_form"):
    trans_date = st.date_input("Transaction Date", defaults.get("trans_date", datetime.today().date()))
    trans_time = st.text_input("Transaction Time (HH:MM, 24h)", defaults.get("trans_time", ""))

    all_categories_ui = list(category_mapping.values())
    default_cat = defaults.get("category", raw_categories[0])
    if default_cat not in all_categories_ui:
        default_cat = all_categories_ui[0]

    category = st.selectbox("Category", all_categories_ui,
                            index=all_categories_ui.index(default_cat))

    amt = st.number_input("Amount", value=float(defaults.get("amt", 0.0)), min_value=0.0)
    gender = st.selectbox("Gender", ["Male", "Female"],
                          index=["Male", "Female"].index(defaults.get("gender", "Male")))

    job_default = defaults.get("job", job_options[0])
    job = st.selectbox("Job (from dataset)", job_options,
                       index=job_options.index(job_default) if job_default in job_options else 0)

    lat = st.number_input("Latitude", value=float(defaults.get("lat", 0.0)))
    long = st.number_input("Longitude", value=float(defaults.get("long", 0.0)))
    city_pop = st.number_input("City Population", value=int(defaults.get("city_pop", 0)))
    merch_lat = st.number_input("Merchant Latitude", value=float(defaults.get("merch_lat", 0.0)))
    merch_long = st.number_input("Merchant Longitude", value=float(defaults.get("merch_long", 0.0)))

    submit_btn = st.form_submit_button("🔍 Predict Fraud")

# -------------------------------
# Prediction
# -------------------------------
if submit_btn:
    try:
        time_obj = datetime.strptime(trans_time, "%H:%M")
    except:
        st.error("⏰ Please enter time in HH:MM format (24h).")
        st.stop()

    trans_datetime = datetime.combine(trans_date, time_obj.time())

    entry = {
        "category": inverse_category_mapping[category],
        "amt": amt,
        "gender": "M" if gender == "Male" else "F",
        "lat": lat,
        "long": long,
        "city_pop": city_pop,
        "job": job,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "hour": trans_datetime.hour,
        "day": trans_datetime.day,
        "month": trans_datetime.month,
        "dayofweek": trans_datetime.weekday()
    }

    df = pd.DataFrame([entry])

    # Encode categoricals
    for col in ["category", "gender", "job"]:
        if col in encoders:
            le = encoders[col]
            mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
            df[col] = df[col].map(mapping).fillna(-1).astype(int)

    # Scale amount
    if "amt" in df.columns:
        df[["amt"]] = scaler.transform(df[["amt"]])

    # Reorder
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Predict
    pred_proba = model.predict_proba(df)[0][1]
    pred = 1 if pred_proba >= 0.5 else 0

    st.markdown("---")
    st.subheader("🔎 Prediction Result")

    col1, col2 = st.columns(2)
    if pred == 1:
        col1.error("⚠️ **Fraud Detected!**")
    else:
        col1.success("✅ **Not Fraud**")

    col2.metric(label="Fraud Probability", value=f"{pred_proba:.2%}")
