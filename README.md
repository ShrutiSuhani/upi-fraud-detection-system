# 🛡️ Real-Time UPI Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-success.svg)

An real-time UPI Fraud Detection System that utilizes a **Defense-in-Depth Hybrid Architecture**. This project combines high-speed deterministic rule engines with an Undersampled Random Forest Classifier to catch complex behavioral anomalies in financial transactions.

---

## 🎯 The Core Problem: The Accuracy Paradox
In real-world financial networks, fraud is incredibly rare but highly damaging. This project was trained on a dataset of 250,000 real UPI transactions where only 480 were fraudulent—an extreme **Class Imbalance Ratio of 520:1**.

Standard machine learning approaches fail here. A naive model guessing "Legitimate" 100% of the time achieves **99.8% Accuracy**, but catches zero fraud (0% Recall). 

**The Solution:**
To break this Accuracy Paradox, this system utilizes **Extreme Undersampling**. By reducing the legitimate class to perfectly match the 480 fraud cases, the Random Forest model was forced to map the actual mathematical boundaries of fraudulent behavior, elevating the F1-Score from 0% to the 46–56% range.

---

## 🏛️ System Architecture
Because purely statistical ML models can struggle with definitive, high-speed attacks (like transaction replays), this system routes every transaction through a 4-Layer Pipeline:

1. **UPI Validation Layer (Deterministic):** Instantly blocks malformed handles or spoofing attempts.
2. **Idempotency Engine (Stateful):** Rejects duplicate Transaction IDs to prevent replay attacks.
3. **Velocity & Cohort Engine (Statistical):** Dynamically compares transaction amounts to the $Mean + 3\sigma$ of the sender's specific geographical and banking cohort.
4. **Machine Learning Model (Behavioral):** An Undersampled Random Forest (100 estimators) evaluates non-linear relationships across 14 dimensions (e.g., Time of Day + Merchant Category + Geography).

---

## 💻 The Risk Operations Dashboard
The system is deployed as a 4-tab interactive web application built with **Streamlit** and **Plotly**.

* **📊 Data Explorer:** Interactive log-scale visualizations exposing the 520:1 class imbalance and transaction patterns.
* **🤖 Model Introspection:** Telemetry displaying the Confusion Matrix, Algorithm Comparison, and Feature Importance (Transaction Amount and Hour identified as primary drivers).
* **🔮 Live Scanner:** A real-time transaction simulator that outputs dynamic risk vectors and transparent alerts showing exactly which layer of the hybrid engine triggered a block.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Engineering:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest, GridSearchCV), imbalanced-learn
* **Frontend UI:** Streamlit
* **Visualizations:** Plotly, Seaborn, Matplotlib
* **Serialization:** Joblib

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/upi-fraud-detection-system.git](https://github.com/your-username/upi-fraud-detection-system.git)
   cd upi-fraud-detection-system
