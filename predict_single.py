import pandas as pd
import joblib, os, json
from datetime import datetime

# -------------------------------
# 1. Load model, scaler, encoders, and feature order
# -------------------------------
model = joblib.load("training_results/fraud_detection_model.pkl")
scaler = joblib.load("training_results/scaler.pkl")
encoders = joblib.load("training_results/encoders.pkl")
feature_columns = joblib.load("training_results/feature_columns.pkl")

# -------------------------------
# 2. Manual transaction entry
# -------------------------------
entry = {
    "trans_date_trans_time": "2019-07-13 12:11",
    "category": "grocery_pos",
    "amt": 288.76,
    "gender": "M",
    "job": "Garment/textile technologist",
    "lat": 41.4798,
    "long": -79.9403,
    "city_pop": 1102,
    "merch_lat": 42.256031,
    "merch_long": -80.074339
}

# -------------------------------
# 3. Process datetime
# -------------------------------
df = pd.DataFrame([entry])
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], format="%Y-%m-%d %H:%M")
df["hour"] = df["trans_date_trans_time"].dt.hour
df["day"] = df["trans_date_trans_time"].dt.day
df["month"] = df["trans_date_trans_time"].dt.month
df["dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
df = df.drop("trans_date_trans_time", axis=1)

# -------------------------------
# 4. Encode categoricals with fallback for unknowns
# -------------------------------
for col in df.select_dtypes(include=["object"]).columns:
    if col in encoders:
        le = encoders[col]
        mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
        df[col] = df[col].map(mapping).fillna(-1).astype(int)
    else:
        df = df.drop(col, axis=1)

# -------------------------------
# 5. Scale amt
# -------------------------------
if "amt" in df.columns:
    df[["amt"]] = scaler.transform(df[["amt"]])

# -------------------------------
# 6. Reorder columns to match training
# -------------------------------
df = df.reindex(columns=feature_columns, fill_value=0)

# -------------------------------
# 7. Predict
# -------------------------------
prediction = model.predict(df)[0]
proba = model.predict_proba(df)[0][1]

# -------------------------------
# 8. Save results
# -------------------------------
os.makedirs("prediction_result_single", exist_ok=True)

result_text = f"Prediction: {'FRAUD' if prediction == 1 else 'NOT FRAUD'}\nFraud probability: {round(proba,4)}"
print(result_text)

with open("prediction_result_single/result.txt", "w") as f:
    f.write(result_text)

result_json = {
    "prediction": int(prediction),
    "prediction_label": "FRAUD" if prediction == 1 else "NOT FRAUD",
    "fraud_probability": round(float(proba), 4)
}
with open("prediction_result_single/result.json", "w") as f:
    json.dump(result_json, f, indent=4)

pd.DataFrame([result_json]).to_csv("prediction_result_single/result.csv", index=False)

print("\n📂 Results saved in 'prediction_result_single/'")
