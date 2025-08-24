

# 💳 Fraud Detection System

A Machine Learning–powered **Credit Card Fraud Detection App** built with **Streamlit**, trained on transaction data, and deployed on **Streamlit Cloud**.

The app allows users to:

* Manually enter transaction details  
* Autofill **Fraud**, **Not Fraud**, or **Random** examples from the dataset  
* Predict whether a transaction is **Fraudulent or Legitimate** in real-time  

---

## 🚀 Demo

👉 [Live App on Streamlit](https://creditcardfd.streamlit.app/)

---

## 📂 Project Structure

```

fraud-detection-app/
│── app.py                  # Streamlit UI app
│── train\_model.py          # Script to train model
│── predict\_single.py       # Script to test single transaction
│── evaluate\_on\_test.py     # Evaluate on test dataset
│── requirements.txt        # Python dependencies
│── fraudSample.csv         # Small dataset sample (for demo)
│── training\_results/       # Saved encoders & feature columns
│     ├── scaler.pkl
│     ├── encoders.pkl
│     └── feature\_columns.pkl

````

> ⚠️ **Note:** The trained model file (`fraud_detection_model.pkl`) is too large for GitHub.  
It is **automatically downloaded from Google Drive** when you run the app.

📥 Manual Download (if needed):  
[fraud_detection_model.pkl (Google Drive)](https://drive.google.com/file/d/1Hz1DMtKSaIFbJe1aUg3ZWf4buv-69jR0/view?usp=sharing)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/fraud-detection-app.git
cd fraud-detection-app
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run app.py
```

---

## 🧠 Model Details

* **Algorithm:** Random Forest Classifier
* **Features used:**

  * Transaction amount, category, location, merchant details
  * Customer demographic info (gender, job, city population)
  * Date-time features (hour, day, month, weekday)
* **Encodings:** Label Encoding for categorical features
* **Scaling:** StandardScaler for transaction amount

---

## 📊 Dataset

This project is based on the **Synthetic Credit Card Fraud Dataset** from [Kaggle](https://www.kaggle.com/code/youssefelbadry10/credit-card-fraud-detection/input).
Due to size limits, only a smaller **fraudSample.csv** is included.

---

## 🌐 Deployment

To deploy on **Streamlit Community Cloud**:

1. Push this repo to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io).
3. Select the repo → choose `app.py` → Deploy 🚀

The app will automatically **download the trained model from Google Drive** on first run.

---

## 🙌 Credits

* Dataset: Kaggle Synthetic Fraud Data
* Libraries: Streamlit, Scikit-learn, Pandas, Joblib, gdown

---

