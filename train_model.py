import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, json, os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# 1. Load dataset
# -------------------------------
print("Loading dataset...")
data = pd.read_csv("fraudTrain.csv")
print("Shape of dataset:", data.shape)

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
# 4. Encode categorical variables
# -------------------------------
cat_cols = data.select_dtypes(include=["object"]).columns
print("Encoding categorical columns:", list(cat_cols))

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

# -------------------------------
# 5. Features and target
# -------------------------------
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# -------------------------------
# 6. Scale numerical features
# -------------------------------
scaler = StandardScaler()
if "amt" in X.columns:
    X[["amt"]] = scaler.fit_transform(X[["amt"]])

# -------------------------------
# 7. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 8. Train model (Random Forest)
# -------------------------------
print("Training RandomForest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -------------------------------
# 9. Evaluation
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_text = classification_report(y_test, y_pred)

print("\nModel Evaluation")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report_text)

# -------------------------------
# 10. Save results & artifacts
# -------------------------------
os.makedirs("training_results", exist_ok=True)

# Save text report
with open("training_results/model_evaluation.txt", "w") as f:
    f.write("Model Evaluation\n")
    f.write("================\n")
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report_text)

# Save model, scaler, and encoders
joblib.dump(model, "training_results/fraud_detection_model.pkl")
joblib.dump(scaler, "training_results/scaler.pkl")
joblib.dump(encoders, "training_results/encoders.pkl")
# Save feature column order
joblib.dump(X_train.columns.tolist(), "training_results/feature_columns.pkl")

# Save metrics (JSON)
metrics = {
    "accuracy": accuracy,
    "precision": report_dict["1"]["precision"],
    "recall": report_dict["1"]["recall"],
    "f1_score": report_dict["1"]["f1-score"]
}
with open("training_results/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n✅ Model, scaler, encoders, features and metrics saved to training_results/")

# -------------------------------
# 11. Visualization
# -------------------------------
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
            xticklabels=["Not Fraud", "Fraud"], 
            yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("training_results/confusion_matrix.png")
plt.close()

report_df = pd.DataFrame(report_dict).transpose()
report_df[["precision","recall","f1-score"]].plot(kind="bar", figsize=(10,6))
plt.title("Classification Report Metrics")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("training_results/classification_report.png")
plt.close()

print("📊 Graphs saved in training_results/ folder")
