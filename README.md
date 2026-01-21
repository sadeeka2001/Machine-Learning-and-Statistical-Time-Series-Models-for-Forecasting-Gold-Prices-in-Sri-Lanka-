# Machine-Learning-and-Statistical-Time-Series-Models-for-Forecasting-Gold-Prices-in-Sri-Lanka-

Perfect 👍
Below is a **complete, well-structured `README.md`** for your **Gold Price Forecasting (ARIMA vs SVR)** project.
It is **academic-ready**, **GitHub-professional**, and suitable for **NIBM / Data Science portfolio** use.

You can **copy–paste this directly into GitHub**.

---

# 🪙 Gold Price Forecasting in Sri Lanka

### A Comparative Study of ARIMA and Support Vector Regression (SVR)

---

## 📌 Project Overview

Gold has long been recognized as a **safe-haven asset and hedge against economic uncertainty**, especially in Sri Lanka where it holds strong cultural and economic value.
This project focuses on **forecasting daily gold prices in Sri Lankan Rupees (LKR)** using both:

* **Statistical Time Series Models**
* **Machine Learning Techniques**

Specifically, the study compares the performance of **ARIMA** and **Support Vector Regression (SVR with RBF kernel)** to determine the most reliable model for short-term gold price forecasting.

---

## 🎯 Research Objectives

* To forecast daily gold prices in LKR using historical data from 2018 onwards
* To compare **ARIMA** and **SVR** based on predictive accuracy
* To identify the most suitable model for short-run gold price prediction in Sri Lanka

---

## 🧠 Models Implemented

### 1️⃣ ARIMA (AutoRegressive Integrated Moving Average)

* Traditional statistical time series model
* Requires stationarity (tested using ADF test)
* Captures linear trends and temporal dependencies
* Model order selected using AIC/BIC criteria

### 2️⃣ Support Vector Regression (SVR – RBF Kernel)

* Machine learning regression model
* Captures **nonlinear and complex patterns**
* Uses lag-based features and normalization
* Hyperparameters tuned using **GridSearchCV (5-fold cross-validation)**

---

## 📊 Dataset

* **Source:** Central Bank of Sri Lanka (API)
* **Frequency:** Daily
* **Time Period:** From 2018 onwards
* **Target Variable:** Gold Price (LKR)

---

## 🔄 Methodology Workflow

1. **Data Collection**

   * Extracted daily gold prices from official sources

2. **Data Preprocessing**

   * Handling missing values using interpolation
   * Normalization using MinMaxScaler
   * Train-test split (80% / 20%)

3. **Stationarity Testing**

   * Augmented Dickey-Fuller (ADF) test
   * First-order differencing applied when required

4. **Feature Engineering**

   * Lag-based sliding window features for SVR
   * Time series structuring for ARIMA

5. **Model Development**

   * ARIMA model selection using auto-arima logic
   * SVR with RBF kernel and optimized hyperparameters

6. **Model Evaluation**

   * Mean Absolute Error (MAE)
   * Mean Absolute Percentage Error (MAPE)
   * Root Mean Squared Error (RMSE)
   * R² Score

---

## 📈 Model Performance Summary

| Metric   | ARIMA | SVR      |
| -------- | ----- | -------- |
| MAE      | High  | Very Low |
| MAPE     | High  | Very Low |
| RMSE     | High  | Very Low |
| R² Score | Low   | ~0.99    |

🔹 **Result:**
SVR significantly outperforms ARIMA across all evaluation metrics, demonstrating superior ability to model nonlinear gold price behavior.

---

## 🛠️ Tech Stack

* **Python**
* **Pandas & NumPy**
* **Scikit-learn**
* **Statsmodels**
* **Matplotlib / Seaborn**
* **Jupyter Notebook / Google Colab**

---

## 📂 Project Structure

```
├── data/
│   └── gold_price_lkr.csv
├── preprocessing.py
├── arima_model.py
├── svr_model.py
├── evaluation.py
├── notebooks/
│   └── analysis.ipynb
├── requirements.txt
└── README.md
```


## 🔮 Future Enhancements

* Incorporate macroeconomic indicators (inflation, exchange rates)
* Develop hybrid ARIMA–SVR models
* Apply deep learning models (LSTM, GRU)
* Deploy forecasts using a web dashboard (Streamlit)

---


##  Conclusion

This study demonstrates that **machine learning models, particularly SVR**, provide more accurate and reliable gold price forecasts than traditional ARIMA models in the Sri Lankan context. The findings highlight the growing importance of ML techniques in financial forecasting for emerging economies.

---


