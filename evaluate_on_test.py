import pandas as pd
import joblib, json, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# 1. Load test dataset
# -------------------------------
print("Loading test dataset...")
data = pd.read_csv("fraudTest.csv")
print("Shape of test dataset:", data.shape)

# -------------------------------
# 2. Drop high-cardinality columns
# -------------------------------
drop_cols = ["Unnamed: 0", "trans_num", "merchant", "first", "last",
             "street", "city", "state", "zip", "dob", "unix_time"]
data = data.drop(columns=[c for c in drop_cols if c in data.columns])

# -------------------------------
# 3. Handle datetime column
# -------------------------------
if "trans_date_trans_time" in data.columns:
    print("Processing datetime column...")
    data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"])
    data["hour"] = data["trans_date_trans_time"].dt.hour
    data["day"] = data["trans_date_trans_time"].dt.day
    data["month"] = data["trans_date_trans_time"].dt.month
    data["dayofweek"] = data["trans_date_trans_time"].dt.dayofweek
    data = data.drop("trans_date_trans_time", axis=1)

# -------------------------------
# 4. Load trained model, scaler, encoders & feature order
# -------------------------------
print("Loading trained model and encoders from training_results/ ...")
model = joblib.load("training_results/fraud_detection_model.pkl")
scaler = joblib.load("training_results/scaler.pkl")
encoders = joblib.load("training_results/encoders.pkl")
feature_columns = joblib.load("training_results/feature_columns.pkl")

# -------------------------------
# 5. Encode categorical variables with saved encoders (robust)
# -------------------------------
cat_cols = data.select_dtypes(include=["object"]).columns
for col in cat_cols:
    if col in encoders:
        le = encoders[col]
        mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
        data[col] = data[col].map(mapping).fillna(-1).astype(int)
    else:
        print(f"⚠️ Column {col} not in encoders, dropping it")
        data = data.drop(col, axis=1)

# -------------------------------
# 6. Preprocess test data
# -------------------------------
X_test = data.drop("is_fraud", axis=1)
y_test = data["is_fraud"]

if "amt" in X_test.columns:
    X_test[["amt"]] = scaler.transform(X_test[["amt"]])

# ✅ Reorder columns to match training
X_test = X_test.reindex(columns=feature_columns, fill_value=0)

# -------------------------------
# 7. Evaluate on test set
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_text = classification_report(y_test, y_pred)

print("\n✅ Evaluation on fraudTest.csv")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report_text)

# -------------------------------
# 8. Save results in test_results/
# -------------------------------
os.makedirs("test_results", exist_ok=True)

# Save text report
with open("test_results/evaluation.txt", "w") as f:
    f.write("Evaluation on fraudTest.csv\n")
    f.write("===========================\n")
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report_text)

# Save metrics JSON
metrics = {
    "accuracy": accuracy,
    "precision": report_dict["1"]["precision"],
    "recall": report_dict["1"]["recall"],
    "f1_score": report_dict["1"]["f1-score"]
}
with open("test_results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save confusion matrix heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Not Fraud", "Fraud"],
            yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Set)")
plt.savefig("test_results/confusion_matrix.png")
plt.close()

# Save classification report bar chart
report_df = pd.DataFrame(report_dict).transpose()
report_df[["precision","recall","f1-score"]].plot(kind="bar", figsize=(10,6))
plt.title("Classification Report Metrics (Test Set)")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("test_results/classification_report.png")
plt.close()

print("\n📂 Results saved in 'test_results/' folder")
