# Banking Fraud Detection 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Library](https://img.shields.io/badge/Library-TensorFlow%2FPyTorch-orange) ![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
This project implements an unsupervised anomaly detection system to identify fraudulent banking transactions. Unlike traditional supervised approaches that struggle with extreme class imbalance, this model utilizes **Deep Autoencoders** to learn a compressed representation of "normal" transaction behavior.

The model flags transactions with high **reconstruction error** as potential fraud, effectively detecting anomalies without needing a balanced dataset.

## 📂 Dataset & Split Strategy
* **Total Volume:** 6 Million transaction rows.
* **Split Strategy:** Strictly **Time-Based Split** to prevent data leakage.
    * **Training:** Historical data (Days 1–25).
    * **Testing:** Future data (Days 26–30).
* **Class Imbalance:** Extreme imbalance (typical of fraud datasets), handled via the unsupervised nature of the Autoencoder.

## 🛠 Feature Engineering
Raw transaction data was enriched using temporal and velocity-based features to capture spending patterns:
* **Time Features:** Hour of day, day of week, is_weekend.
* **Velocity Checks:** Number of transactions in the last 1hr, 24hr, and 7-day windows.
* **Aggregations:** Average transaction amount per customer over rolling windows.
* **Ratios:** Ratio of current transaction amount to the customer's average spending (to detect spikes).

## 🧠 Model Architecture: Autoencoder
The core model is a Deep Autoencoder (Neural Network) trained **only on non-fraudulent (majority) transactions**.

1.  **Encoder:** Compresses the 40+ input features into a lower-dimensional latent space.
2.  **Bottleneck:** Forces the model to learn the most essential patterns of legitimate spending.
3.  **Decoder:** Attempts to reconstruct the original input from the compressed representation.
4.  **Anomaly Detection:**
    * When the model sees a **Normal** transaction, the reconstruction error is low.
    * When the model sees **Fraud**, it fails to reconstruct the pattern effectively, resulting in a **high reconstruction error**.

## 📊 Results (Future Data Evaluation)
The model was evaluated on "Unseen Future Data" (Days 26-30).

**Confusion Matrix Metrics:**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Legit)** | 0.99 | 1.00 | 1.00 | 101,973 |
| **1 (Fraud)** | **0.76** | **0.54** | **0.63** | 1,600 |
| **Accuracy** | | | **0.99** | 103,573 |

### Analysis
* **Precision (0.76):** When the model flags a transaction as fraud, it is correct 76% of the time. This is a strong result for a fraud system, minimizing customer friction (False Positives).
* **Recall (0.54):** The model successfully caught 54% of all fraud cases in the future dataset. While lower than the precision, this is significant given the unsupervised nature of the approach.
* **Stability:** The model maintained performance on future days, indicating it has not overfit to specific past timeframes.

## 🚀 How to Run
1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Feature Engineering:**
    ```bash
    python src/feature_engineering.py --input data/raw.csv
    ```
3.  **Train Autoencoder:**
    ```bash
    python src/train_autoencoder.py --epochs 50
    ```
4.  **Evaluate:**
    ```bash
    python src/evaluate.py --threshold auto
    ```

## 🔮 Future Improvements
* **Hybrid Approach:** Use the Reconstruction Error from the Autoencoder as a *feature* input into a supervised XGBoost model to improve Recall.
* **Threshold Tuning:** Implement dynamic thresholding based on time of day to catch more fraud during peak hours.
