# CHAPTER 4: IMPLEMENTATION AND RESULTS

---

## 4.1 INTRODUCTION

This chapter presents the implementation of a fraud detection system using machine learning
techniques applied to credit card transaction data. The primary objective is to evaluate and
compare the performance of three classification models — a Neural Network, Logistic
Regression, and Naive Bayes — under two experimental conditions: training on the original
imbalanced dataset and training on a balanced dataset produced using the Synthetic Minority
Over-sampling Technique (SMOTE).

The chapter is structured as follows. Section 4.2 describes the dataset and the preprocessing
steps applied to prepare the data for modelling. Section 4.3 covers the implementation of each
model, including architecture decisions, training procedures, and evaluation metrics. Section 4.4
discusses the deployment of the trained models through an interactive dashboard. The results
and findings are discussed throughout each section.

---

## 4.2 DATA EXPLORATION AND PREPROCESSING

### 4.2.1 Initial Data Overview

Two datasets were used in this project:

**Dataset 1 — Credit Card Transactions (EDA / Autoencoder)**
This dataset contains 46,334 real-world credit card transactions with 24 columns, including
transaction metadata such as merchant name, category, transaction amount, cardholder
location, and a binary fraud label (`is_fraud`). After removing irrelevant identifier columns
(e.g., cardholder name, card number, transaction ID), 14 features were retained for modelling.

**Dataset 2 — Anonymised Credit Card Transactions (Supervised Models)**
This dataset contains 284,807 transactions with 31 columns. Features V1 through V28 are
the result of a Principal Component Analysis (PCA) transformation applied to protect
cardholder privacy. The remaining features are `Time`, `Amount`, and `Class` (0 = normal,
1 = fraud). This dataset was used to train and evaluate the Neural Network, Logistic
Regression, and Naive Bayes classifiers.

---

*[INSERT TABLE 1: Dataset Summary — Rows, Columns, Fraud Rate]*

| Dataset | Rows | Columns | Fraud Cases | Fraud Rate |
|---|---|---|---|---|
| Credit Card Transactions (EDA) | 46,334 | 24 (raw) / 14 (cleaned) | 424 | 0.92% |
| Anonymised Transactions (Models) | 284,807 | 31 | 492 | 0.17% |

---

### 4.2.2 Importing Libraries

The following Python libraries were used throughout the project:

- **pandas** and **numpy** — data loading, manipulation, and numerical computation
- **matplotlib** and **seaborn** — data visualisation
- **scikit-learn** — preprocessing, model training, and evaluation metrics
- **imbalanced-learn** — SMOTE oversampling
- **TensorFlow / Keras** — Neural Network construction and training

---

*[INSERT FIGURE 1: Code snippet showing library imports]*

---

### 4.2.3 Loading and Inspecting the Dataset

The anonymised dataset was loaded using `pandas.read_csv()`. An initial inspection confirmed
the dataset contains 284,807 rows and 31 columns, with no missing values. The class
distribution revealed a severe imbalance: 284,315 normal transactions (99.83%) versus only
492 fraudulent transactions (0.17%).

---

*[INSERT FIGURE 2: Output of `df.shape` and `df.info()` — showing 284,807 rows, 31 columns]*

*[INSERT FIGURE 3: Class distribution bar chart — fraud_vs_normal_transactions.png]*

---

The extreme imbalance is a critical challenge for fraud detection. A naive classifier that
predicts "normal" for every transaction would achieve 99.83% accuracy while detecting zero
fraud cases. This motivates the use of SMOTE to balance the training data.

### 4.2.4 Exploratory Data Analysis

Several visualisations were produced to understand the distribution of features and the
relationship between transaction characteristics and fraud.

**Transaction Amount Distribution**

The transaction amount is heavily right-skewed, with most transactions below $200 and a
long tail extending to over $25,000. This skewness is typical of financial data.

---

*[INSERT FIGURE 4: Transaction Amount Distribution — transaction_amount_distribution.png]*

---

**Fraud vs. Transaction Amount**

A box plot comparing transaction amounts for fraudulent and normal transactions shows that
fraudulent transactions tend to have lower median amounts than normal transactions, though
there is overlap. This suggests that transaction amount alone is not a reliable fraud indicator.

---

*[INSERT FIGURE 5: Fraud vs. Amount Box Plot — fraud_vs_amount_boxplot.png]*

---

**Transaction Time Distribution**

The time distribution shows two distinct peaks, suggesting higher transaction activity during
certain periods of the day. The `Time` feature was retained as a numerical input after scaling.

---

*[INSERT FIGURE 6: Transaction Time Distribution — transaction_time_distribution.png]*

---

**Feature Correlation Heatmap**

Because features V1–V28 are PCA-transformed, they are largely uncorrelated with each
other. The heatmap confirms this, with most off-diagonal values close to zero.

---

*[INSERT FIGURE 7: Feature Correlation Heatmap — correlation_heatmap.png]*

---

### 4.2.5 Data Preprocessing

The following preprocessing steps were applied before model training:

1. **Feature and Target Separation** — The `Class` column was separated as the target
   variable `y`. All remaining columns formed the feature matrix `X`.

2. **Train-Test Split** — The dataset was split into 80% training (227,845 samples) and
   20% test (56,962 samples) using stratified sampling to preserve the class ratio.

3. **Feature Scaling** — `StandardScaler` was applied to the `Amount` and `Time` columns
   to normalise their ranges. The scaler was fitted on the training set and applied to the
   test set to prevent data leakage.

4. **SMOTE Balancing** — For the balanced experiments, SMOTE was applied to the
   training set to generate synthetic fraud samples, producing a balanced training set of
   227,451 samples per class.

---

*[INSERT FIGURE 8: Class distribution after SMOTE — balanced_training_data_smote.png]*

*[INSERT TABLE 2: Train/Test Split Summary]*

| Split | Total Samples | Normal | Fraud |
|---|---|---|---|
| Training (Imbalanced) | 227,845 | 227,451 | 394 |
| Training (Balanced — SMOTE) | 454,902 | 227,451 | 227,451 |
| Test Set | 56,962 | 56,864 | 98 |

---

## 4.3 MODEL IMPLEMENTATION AND EVALUATION

Three models were implemented and evaluated: a Neural Network, Logistic Regression, and
Naive Bayes. Each model was trained twice — once on the imbalanced dataset and once on
the SMOTE-balanced dataset — and evaluated on the same held-out test set.

### 4.3.1 Neural Network

**Architecture**

A fully connected feedforward neural network was constructed using TensorFlow/Keras. The
architecture consists of:

- Input layer: 30 features (V1–V28, Time, Amount)
- Hidden layers: Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU)
- Output layer: Dense(1, Sigmoid) — binary classification

The model was compiled with the Adam optimiser and binary cross-entropy loss. Training was
performed for 20 epochs with a batch size of 256 and a 20% validation split.

---

*[INSERT FIGURE 9: Neural Network Training Loss Curve — loss_curve.png]*

---

**Results — Imbalanced Dataset**

When trained on the imbalanced dataset, the Neural Network achieved 100% weighted
accuracy but a Recall of 0% for the fraud class. The model learned to predict "normal" for
all transactions, exploiting the class imbalance.

**Results — Balanced Dataset (SMOTE)**

After SMOTE balancing, the Neural Network achieved a Recall of 85% and a Precision of
63% for the fraud class, with a ROC-AUC of 0.97. This represents a substantial improvement
in fraud detection capability.

---

*[INSERT FIGURE 10: Confusion Matrix — Neural Network (Imbalanced) — confusion_matrix_nn_imbalanced.png]*

*[INSERT FIGURE 11: Confusion Matrix — Neural Network (Balanced) — confusion_matrix_nn_balanced.png]*

*[INSERT FIGURE 12: ROC Curve — Neural Network (Imbalanced) — roc_curve_imbalanced.png]*

*[INSERT FIGURE 13: ROC Curve — Neural Network (Balanced) — roc_curve_balanced.png]*

*[INSERT FIGURE 14: Precision–Recall vs. Threshold — precision_recall_threshold.png]*

---

### 4.3.2 Logistic Regression

**Implementation**

Logistic Regression was implemented using scikit-learn's `LogisticRegression` class with a
maximum of 5,000 iterations to ensure convergence. The `liblinear` solver was used for the
imbalanced baseline, and the default solver was used for the SMOTE-balanced experiment.

**Results — Imbalanced Dataset**

The imbalanced Logistic Regression model achieved 100% weighted accuracy, a Precision of
83%, and a Recall of 65% for the fraud class. This is the best precision achieved across all
imbalanced models, indicating that when the model does flag a transaction as fraud, it is
correct 83% of the time.

**Results — Balanced Dataset (SMOTE)**

After SMOTE balancing, Recall improved to 90% — the highest recall across all models —
but Precision dropped to 13%, meaning a large number of false alarms. The ROC-AUC
remained at 0.97.

---

*[INSERT FIGURE 15: Confusion Matrix — Logistic Regression (Imbalanced) — confusion_matrix_lr_imbalanced.png]*

*[INSERT FIGURE 16: Confusion Matrix — Logistic Regression (Balanced) — confusion_matrix_lr_balanced.png]*

---

### 4.3.3 Naive Bayes

**Implementation**

A Gaussian Naive Bayes classifier was implemented using scikit-learn's `GaussianNB` class.
Naive Bayes assumes feature independence and models each feature as a Gaussian
distribution. It is computationally efficient and serves as a fast baseline.

**Results — Imbalanced Dataset**

The imbalanced Naive Bayes model achieved 99% weighted accuracy, a Precision of 14%,
and a Recall of 66% for the fraud class, with a ROC-AUC of 0.97.

**Results — Balanced Dataset (SMOTE)**

After SMOTE balancing, Recall improved to 88%, but Precision dropped to 5% — the lowest
precision across all models. This indicates a high false positive rate.

---

*[INSERT FIGURE 17: Confusion Matrix — Naive Bayes (Imbalanced) — confusion_matrix_nb_imbalanced.png]*

---

### 4.3.4 Model Accuracy Comparison

The table below summarises the performance of all models on the held-out test set.

---

*[INSERT TABLE 3: Model Performance Summary]*

| Model | Data Type | Accuracy | Precision (Fraud) | Recall (Fraud) | ROC-AUC |
|---|---|---|---|---|---|
| Neural Network | Imbalanced | 100% | 11% | 0% | 0.50 |
| Neural Network | Balanced (SMOTE) | 100% | 63% | 85% | 0.97 |
| Logistic Regression | Imbalanced | 100% | 83% | 65% | 0.97 |
| Logistic Regression | Balanced (SMOTE) | 99% | 13% | 90% | 0.97 |
| Naive Bayes | Imbalanced | 99% | 14% | 66% | 0.97 |
| Naive Bayes | Balanced (SMOTE) | 99% | 5% | 88% | 0.96 |

---

*[INSERT FIGURE 18: Bar Chart — Precision Comparison (Imbalanced vs. Balanced)]*

*[INSERT FIGURE 19: Bar Chart — Recall Comparison (Imbalanced vs. Balanced)]*

*[INSERT FIGURE 20: Bar Chart — ROC-AUC Comparison (Imbalanced vs. Balanced)]*

---

**Key Observations:**

- Accuracy is a misleading metric for imbalanced fraud detection. All models achieve
  near-perfect accuracy on the imbalanced dataset, yet the Neural Network catches zero fraud.
- SMOTE balancing dramatically improves Recall across all models, at the cost of lower
  Precision (more false alarms).
- The Neural Network (Balanced) offers the best trade-off between Precision (63%) and
  Recall (85%), making it the most suitable model for production deployment.
- Logistic Regression (Balanced) achieves the highest Recall (90%) and is a strong,
  interpretable baseline.
- Naive Bayes is the fastest model but has the lowest Precision (5% balanced), making it
  unsuitable for production without threshold tuning.

### 4.3.5 Model Loss (Neural Network)

The training and validation loss curves for the Neural Network are shown below. The loss
decreases steadily over 20 epochs, indicating that the model is learning effectively without
significant overfitting.

---

*[INSERT FIGURE 21: Neural Network Training and Validation Loss — loss_curve.png]*

---

### 4.3.6 ROC Curve Comparison

The ROC curves for all models are plotted together below. Models trained on the balanced
dataset (solid lines) consistently outperform their imbalanced counterparts (dashed lines).
The Neural Network trained on the imbalanced dataset has an AUC of 0.50, equivalent to
random guessing, confirming that it learned no useful patterns for fraud detection.

---

*[INSERT FIGURE 22: ROC Curve Comparison — All Models — roc_curve_comparison.png]*

---

## 4.4 MODEL DEPLOYMENT

The trained models were deployed as an interactive web dashboard built using the Streamlit
framework. The dashboard provides a user-friendly interface for exploring model performance
and understanding the impact of class imbalance on fraud detection.

**Dashboard Features:**

1. **Dataset Overview** — Displays class distribution, transaction amount distribution,
   time distribution, and feature correlation heatmap.

2. **Model Performance Table** — A comparative table showing Accuracy, Precision, Recall,
   and ROC-AUC for all six model/dataset combinations.

3. **Model Comparison Charts** — Side-by-side bar charts comparing Precision, Recall, and
   ROC-AUC across all models for both imbalanced and balanced conditions.

4. **Confusion Matrix Explorer** — An interactive dropdown allowing users to select any
   model and dataset type to view the corresponding confusion matrix alongside key metrics.

5. **ROC Curve Comparison** — Individual and combined ROC curves for all models.

6. **Neural Network Training Visualisations** — Loss curves, SMOTE data distribution,
   PCA feature distribution, and Precision–Recall vs. Threshold charts.

7. **Key Insights** — A summary of findings and model recommendations.

8. **Threshold Tuning** — An interactive slider allowing users to adjust the classification
   threshold and observe the resulting changes in Precision, Recall, and the confusion matrix.

The dashboard is accessible locally by running:

```
streamlit run app/dashboard.py
```

---

*[INSERT FIGURE 23: Screenshot of Dashboard — Home Page]*

*[INSERT FIGURE 24: Screenshot of Dashboard — Confusion Matrix Explorer]*

*[INSERT FIGURE 25: Screenshot of Dashboard — Threshold Tuning Page]*

---
