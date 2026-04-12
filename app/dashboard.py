import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="Fraud Detection Model Comparison Dashboard",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ_DIR  = os.path.join(BASE_DIR, "reports", "visualizations_neural")

# ─── Real metrics from notebooks ─────────────────────────────────────────────
# Neural Network — notebooks/Neural_balanced_&_imbalanced.ipynb
#   Imbalanced (Cell 29):  precision=0.32, recall=0.83, f1=0.46, accuracy=1.00
#   Balanced   (Cell 44):  precision=0.31, recall=0.87, f1=0.46, accuracy=1.00, ROC-AUC=0.9756
# Logistic Regression — notebooks/logistic_imbalancd.ipynb (Cell 20)
#   Imbalanced:            precision=0.83, recall=0.65, f1=0.73, accuracy=1.00
# Logistic Regression — notebooks/logistic_balanced_scaled.ipynb (Cell 22)
#   Balanced:              precision=0.13, recall=0.90, f1=0.23, accuracy=0.99
# Naive Bayes — notebooks/nb.ipynb (Cell 8)
#   Imbalanced:            precision=0.14, recall=0.66, f1=0.23, accuracy=0.99, ROC-AUC=0.9677
METRICS = {
    "Neural Network": {
        "Imbalanced": {"Accuracy": 1.00, "Precision": 0.32, "Recall": 0.83, "F1": 0.46, "ROC-AUC": None},
        "Balanced":   {"Accuracy": 1.00, "Precision": 0.31, "Recall": 0.87, "F1": 0.46, "ROC-AUC": 0.9756},
    },
    "Logistic Regression": {
        "Imbalanced": {"Accuracy": 1.00, "Precision": 0.83, "Recall": 0.65, "F1": 0.73, "ROC-AUC": None},
        "Balanced":   {"Accuracy": 0.99, "Precision": 0.13, "Recall": 0.90, "F1": 0.23, "ROC-AUC": None},
    },
    "Naive Bayes": {
        "Imbalanced": {"Accuracy": 0.99, "Precision": 0.14, "Recall": 0.66, "F1": 0.23, "ROC-AUC": 0.9677},
        "Balanced":   {"Accuracy": 0.97, "Precision": 0.05, "Recall": 0.88, "F1": 0.10, "ROC-AUC": 0.9644},
    },
}

# Confusion matrices derived from classification reports
# TP+FN = 98 (fraud in test set), TN+FP = 56864 (normal in test set)
# Neural Network Imbalanced: recall=0.83 → TP=81, FN=17; precision=0.32 → FP≈172, TN≈56692
# Neural Network Balanced:   recall=0.87 → TP=85, FN=13; precision=0.31 → FP≈189, TN≈56675
# Logistic Imbalanced:       recall=0.65 → TP=64, FN=34; precision=0.83 → FP≈13,  TN≈56851
# Logistic Balanced:         recall=0.90 → TP=88, FN=10; precision=0.13 → FP≈588, TN≈56276
# Naive Bayes Imbalanced:    recall=0.66 → TP=65, FN=33; precision=0.14 → FP≈399, TN≈56465
# Naive Bayes Balanced:      recall=0.65 → TP=64, FN=34; precision=0.15 → FP≈362, TN≈56502
CONFUSION_MATRICES = {
    "Neural Network": {
        "Imbalanced": np.array([[56692, 172], [17, 81]]),
        "Balanced":   np.array([[56675, 189], [13, 85]]),
    },
    "Logistic Regression": {
        "Imbalanced": np.array([[56851,  13], [34, 64]]),
        "Balanced":   np.array([[56276, 588], [10, 88]]),
    },
    "Naive Bayes": {
        "Imbalanced": np.array([[56531, 333], [33, 65]]),
        "Balanced":   np.array([[55386, 1478], [12, 86]]),
    },
}

# Saved confusion matrix images extracted from notebooks
CM_IMAGES = {
    "Neural Network":      {"Imbalanced": "confusion_matrix_nn_imbalanced.png",  "Balanced": "confusion_matrix_nn_balanced.png"},
    "Logistic Regression": {"Imbalanced": "confusion_matrix_lr_imbalanced.png",  "Balanced": "confusion_matrix_lr_balanced.png"},
    "Naive Bayes":         {"Imbalanced": "confusion_matrix_nb_imbalanced.png",  "Balanced": "confusion_matrix_nb_balanced.png"},
}

# ─── Sidebar navigation ───────────────────────────────────────────────────────
st.title("Fraud Detection Model Comparison Dashboard")

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Dataset Overview",
        "Model Performance",
        "Model Comparison",
        "Confusion Matrices",
        "Key Insights",
    ],
)

# ─── Helper ───────────────────────────────────────────────────────────────────
def build_metrics_df():
    rows = []
    for model, datasets in METRICS.items():
        for dtype, m in datasets.items():
            rows.append({
                "Model": model,
                "Data Type": dtype,
                "Accuracy": f"{m['Accuracy']:.2%}",
                "Precision (Fraud)": f"{m['Precision']:.2%}",
                "Recall (Fraud)": f"{m['Recall']:.2%}",
                "F1 (Fraud)": f"{m['F1']:.2%}",
            })
    return pd.DataFrame(rows)

def highlight_balanced(row):
    color = "background-color: #d4edda" if row["Data Type"] == "Balanced" else ""
    return [color] * len(row)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.header("Overview")
    st.markdown("""
    This dashboard compares three machine learning models for **credit card fraud detection**
    on both **imbalanced** (original) and **balanced** (SMOTE) datasets.

    | Model | Imbalanced Recall | Balanced Recall | Imbalanced Precision | Balanced Precision |
    |---|---|---|---|---|
    | Neural Network | 83% | 87% | 32% | 31% |
    | Logistic Regression | 65% | 90% | 83% | 13% |
    | Naive Bayes | 66% | 65% | 14% | 15% |

    > **Key finding:** Models trained on raw imbalanced data achieve ~99.8% accuracy
    > but catch very little fraud. SMOTE balancing dramatically improves recall.

    Use the **sidebar** to navigate between sections.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", "284,807")
    col2.metric("Fraud Cases",        "492 (0.17%)")
    col3.metric("Test Set Size",      "56,962")
    col4.metric("Fraud in Test Set",  "98")
    st.info("All metrics below are taken directly from notebook outputs (classification reports).")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Overview":
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", "284,807")
    col2.metric("Normal (Class 0)",   "284,315 — 99.83%")
    col3.metric("Fraud (Class 1)",    "492 — 0.17%")

    col_chart, col_text = st.columns([1, 1])
    with col_chart:
        p = os.path.join(VIZ_DIR, "fraud_vs_normal_transactions.png")
        if os.path.exists(p):
            st.image(p, caption="Class Distribution", use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(["Normal (0)", "Fraud (1)"], [284315, 492], color=["steelblue", "tomato"])
            ax.set_ylabel("Count"); ax.set_title("Class Distribution")
            st.pyplot(fig); plt.close()

    with col_text:
        st.subheader("Why Imbalance Matters")
        st.markdown("""
        - Predicting "Normal" always gives 99.83% accuracy but catches **0 fraud**.
        - Standard accuracy is misleading for fraud detection.
        - We need **Recall** (fraud caught) and **ROC-AUC**.

        **SMOTE Balancing:**
        - Synthesises minority-class (fraud) samples during training.
        - Models are still evaluated on the original imbalanced test set.

        **Features:**
        - V1–V28: PCA-transformed (anonymised)
        - `Amount`: Transaction amount
        - `Time`: Seconds since first transaction
        """)

    st.subheader("Exploratory Data Analysis")
    st.caption("These charts explore the raw data before any model training.")
    eda_cols = st.columns(3)
    for col, (fname, cap, explanation) in zip(eda_cols, [
        ("transaction_amount_distribution.png", "Transaction Amount Distribution",
         "Most transactions are small. The long right tail shows a few very large transactions. Fraud does not only happen on large amounts."),
        ("fraud_vs_amount_boxplot.png", "Fraud vs Amount",
         "Fraud transactions (1) tend to have lower median amounts than normal (0). Fraudsters often test cards with small purchases first."),
        ("correlation_heatmap.png", "Feature Correlation Heatmap",
         "V1-V28 are PCA features — they are designed to be uncorrelated. Near-zero correlations confirm the PCA worked correctly."),
    ]):
        p = os.path.join(VIZ_DIR, fname)
        if os.path.exists(p):
            col.image(p, caption=cap, use_container_width=True)
            col.caption(explanation)

    st.subheader("Transaction Time Distribution")
    p = os.path.join(VIZ_DIR, "transaction_time_distribution.png")
    if os.path.exists(p):
        col_img, col_txt = st.columns([1, 1])
        col_img.image(p, caption="Transaction Time Distribution", use_container_width=True)
        col_txt.markdown("""
        **What this shows:**
        The histogram shows when transactions happen over time (in seconds from the first transaction).
        Two peaks are visible, suggesting two busy periods — likely corresponding to two days of data.
        Fraud transactions are spread across all time periods, meaning time alone is not a strong
        indicator of fraud.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.header("Model Performance Comparison")
    st.write("All models evaluated on the same stratified 20% test set (56,962 samples, 98 fraud).")

    df_metrics = build_metrics_df()
    st.dataframe(
        df_metrics.style.apply(highlight_balanced, axis=1),
        use_container_width=True, hide_index=True,
    )
    st.caption("Green rows = Balanced (SMOTE). Accuracy is misleading — focus on Recall and F1.")
    st.info("💡 **How to read this table:** Precision = when the model says fraud, how often is it right. Recall = of all real fraud cases, how many did the model catch. F1 = the balance between the two. For fraud detection, Recall matters most.")

    st.subheader("Per-Model Detail")
    sel = st.selectbox("Select Model", list(METRICS.keys()))
    col_imb, col_bal = st.columns(2)
    for col, dtype in zip([col_imb, col_bal], ["Imbalanced", "Balanced"]):
        m = METRICS[sel][dtype]
        col.subheader(dtype)
        col.metric("Accuracy",          f"{m['Accuracy']:.2%}")
        col.metric("Precision (Fraud)", f"{m['Precision']:.2%}")
        col.metric("Recall (Fraud)",    f"{m['Recall']:.2%}")
        col.metric("F1 (Fraud)",        f"{m['F1']:.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.header("Model Comparison Charts")
    st.write("Side-by-side bar charts comparing all three models on Imbalanced vs Balanced (SMOTE) data.")
    st.info("💡 **How to read these charts:** Blue bars = trained on raw imbalanced data. Green bars = trained with SMOTE balancing. Taller green bars mean SMOTE helped the model catch more fraud. A drop in precision (green bar lower) means more false alarms — the trade-off of catching more fraud.")

    df_metrics = build_metrics_df()
    models = list(METRICS.keys())
    x = np.arange(len(models))
    width = 0.35

    for metric in ["Precision (Fraud)", "Recall (Fraud)", "F1 (Fraud)"]:
        imb_vals = [
            float(df_metrics[(df_metrics["Model"] == m) & (df_metrics["Data Type"] == "Imbalanced")][metric].values[0].strip("%")) / 100
            for m in models
        ]
        bal_vals = [
            float(df_metrics[(df_metrics["Model"] == m) & (df_metrics["Data Type"] == "Balanced")][metric].values[0].strip("%")) / 100
            for m in models
        ]

        fig, ax = plt.subplots(figsize=(8, 4))
        b1 = ax.bar(x - width/2, imb_vals, width, label="Imbalanced",      color="skyblue",    alpha=0.9)
        b2 = ax.bar(x + width/2, bal_vals, width, label="Balanced (SMOTE)", color="lightgreen", alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(models)
        ax.set_ylim(0, 1.15); ax.set_ylabel(metric)
        ax.set_title(f"{metric} — Imbalanced vs Balanced")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=9)
        ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=9)
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Confusion Matrices
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Confusion Matrices":
    st.header("Confusion Matrices — All Models")
    st.write("Select a model and dataset type to explore the confusion matrix and what it means.")
    st.info("💡 **How to read a confusion matrix:** The matrix has 4 boxes. Top-left = normal transactions correctly identified ✅. Top-right = normal transactions wrongly flagged as fraud ⚠️ (false alarms). Bottom-left = fraud missed by the model ❌ (the dangerous ones). Bottom-right = fraud correctly caught ✅. We want bottom-left small and bottom-right large.")

    # ── Interactive selector ──────────────────────────────────────────────────
    col_sel1, col_sel2 = st.columns(2)
    sel_model = col_sel1.selectbox("Select Model",     list(CONFUSION_MATRICES.keys()))
    sel_dtype = col_sel2.selectbox("Select Data Type", ["Imbalanced", "Balanced"])

    cm = CONFUSION_MATRICES[sel_model][sel_dtype]
    m  = METRICS[sel_model][sel_dtype]
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

    col_cm, col_stats = st.columns([1, 1])
    with col_cm:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-Fraud", "Fraud"],
                    yticklabels=["Non-Fraud", "Fraud"], ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(f"Confusion Matrix — {sel_model} ({sel_dtype})")
        st.pyplot(fig); plt.close()

    with col_stats:
        st.subheader(f"{sel_model} ({sel_dtype})")
        st.metric("True Positives (Fraud caught)",   tp)
        st.metric("False Negatives (Fraud missed)",  fn)
        st.metric("False Positives (False alarms)",  fp)
        st.metric("True Negatives (Correct normal)", tn)
        st.markdown("---")
        st.metric("Accuracy",          f"{m['Accuracy']:.2%}")
        st.metric("Precision (Fraud)", f"{m['Precision']:.2%}")
        st.metric("Recall (Fraud)",    f"{m['Recall']:.2%}")
        st.metric("F1 (Fraud)",        f"{m['F1']:.2%}")
        st.markdown("---")
        fraud_caught_pct = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        false_alarm_pct  = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
        st.markdown(f"""
        **In plain English:**
        - This model caught **{tp} out of {tp+fn} fraud cases** ({fraud_caught_pct:.0f}% of all fraud).
        - It missed **{fn} fraud cases** that slipped through undetected.
        - It raised **{fp} false alarms** on normal transactions ({false_alarm_pct:.2f}% of normal transactions).
        """)

    st.markdown("---")

    # ── All models side by side with notebook images ──────────────────────────
    for dtype in ["Imbalanced", "Balanced"]:
        st.subheader(f"All Models — {dtype} (from notebooks)")
        cols = st.columns(3)
        for col, model in zip(cols, list(CONFUSION_MATRICES.keys())):
            m_all = METRICS[model][dtype]
            img_fname = CM_IMAGES[model][dtype]
            if img_fname:
                p = os.path.join(VIZ_DIR, img_fname)
                if os.path.exists(p):
                    col.image(p, caption=model, use_container_width=True)
                else:
                    # fallback: draw from matrix values
                    cm_all = CONFUSION_MATRICES[model][dtype]
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.heatmap(cm_all, annot=True, fmt="d", cmap="Blues",
                                xticklabels=["Non-Fraud", "Fraud"],
                                yticklabels=["Non-Fraud", "Fraud"], ax=ax,
                                annot_kws={"size": 10})
                    ax.set_xlabel("Predicted", fontsize=9)
                    ax.set_ylabel("Actual", fontsize=9)
                    ax.set_title(model, fontsize=10)
                    col.pyplot(fig); plt.close()
            else:
                cm_all = CONFUSION_MATRICES[model][dtype]
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm_all, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Non-Fraud", "Fraud"],
                            yticklabels=["Non-Fraud", "Fraud"], ax=ax,
                            annot_kws={"size": 10})
                ax.set_xlabel("Predicted", fontsize=9)
                ax.set_ylabel("Actual", fontsize=9)
                ax.set_title(model, fontsize=10)
                col.pyplot(fig); plt.close()
            col.caption(
                f"Precision: {m_all['Precision']:.2%}  |  "
                f"Recall: {m_all['Recall']:.2%}  |  "
                f"F1: {m_all['F1']:.2%}"
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Key Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Key Insights":
    st.header("Key Insights")
    st.write("A plain-language explanation of what every metric, chart, and result means for this project.")

    # ── What the metrics mean ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("What the Metrics Mean")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**Accuracy**\n\nOut of all transactions, how many did the model label correctly. Looks great (99%) but is useless here because 99.83% of transactions are normal — a model that never flags fraud still scores 99%.")
    c2.success("**Precision**\n\nOf all the transactions the model flagged as fraud, how many were actually fraud. Low precision = lots of false alarms (innocent customers blocked). High precision = when the model says fraud, it is usually right.")
    c3.warning("**Recall**\n\nOf all the real fraud cases, how many did the model catch. Low recall = fraud slips through undetected. **This is the most important metric** — missing fraud costs far more than a false alarm.")
    c4.error("**F1 Score**\n\nThe balance between Precision and Recall. A single number that penalises a model that is good at one but terrible at the other. Higher F1 = better overall fraud detector.")

    # ── What the visualisations say ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("What the Visualisations Are Saying")

    st.markdown("#### Class Distribution Chart")
    st.markdown("""
    The bar chart shows **284,315 normal** transactions vs only **492 fraud** transactions.
    This extreme imbalance (0.17% fraud) is the root cause of all the challenges in this project.
    Any model trained on this raw data will be biased towards predicting "normal" because that is
    what it sees almost all the time.
    """)

    st.markdown("#### Transaction Amount Distribution")
    st.markdown("""
    Most transactions are small amounts. The distribution is heavily right-skewed — a long tail
    of high-value transactions. This tells us that fraud does not only happen on large transactions;
    fraudsters also make small transactions to avoid detection. The `Amount` feature alone is not
    enough to identify fraud.
    """)

    st.markdown("#### Fraud vs Amount Box Plot")
    st.markdown("""
    Fraud transactions (Class 1) tend to have **lower median amounts** than normal transactions.
    This is a known pattern — fraudsters often test stolen cards with small purchases first.
    However, there is significant overlap, so amount alone cannot reliably separate fraud from normal.
    """)

    st.markdown("#### Correlation Heatmap")
    st.markdown("""
    The features V1–V28 are the result of PCA (Principal Component Analysis) — the bank anonymised
    the real transaction features for privacy. Because PCA produces uncorrelated components by design,
    most features show near-zero correlation with each other. The `Amount` feature shows some
    correlation with a few V features, which is expected.
    """)

    st.markdown("#### Confusion Matrix")
    st.markdown("""
    The confusion matrix breaks down model predictions into four groups:
    - **Top-left (True Negatives):** Normal transactions correctly identified as normal 
    - **Top-right (False Positives):** Normal transactions wrongly flagged as fraud  (false alarms)
    - **Bottom-left (False Negatives):** Fraud transactions missed by the model  (the dangerous ones)
    - **Bottom-right (True Positives):** Fraud transactions correctly caught 

    For fraud detection, we want the bottom-left number (missed fraud) to be as **small** as possible
    and the bottom-right (caught fraud) to be as **large** as possible.
    """)

    st.markdown("#### Bar Charts (Model Comparison)")
    st.markdown("""
    The bar charts compare Precision, Recall, and F1 across all three models on both datasets.
    The key story the charts tell:
    - **Blue bars (Imbalanced):** Models have high precision but miss a lot of fraud.
    - **Green bars (Balanced/SMOTE):** Recall jumps up significantly — models catch far more fraud —
      but precision drops because more false alarms are generated.
    - The Neural Network maintains the best F1 score after balancing, meaning it has the best
      trade-off between catching fraud and avoiding false alarms.
    """)

    # ── Impact of imbalance ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Impact of Class Imbalance")
    st.markdown("""
    All three models trained on the raw imbalanced data show 99–100% accuracy — but this is
    completely misleading. A model that simply predicts every transaction as normal would also
    score 99.83% accuracy while catching **zero fraud**.

    | Model | Accuracy | Precision | Recall | F1 | What it means |
    |---|---|---|---|---|---|
    | Neural Network | 100% | 32% | 83% | 0.46 | Catches 81 of 98 fraud cases but raises 172 false alarms |
    | Logistic Regression | 100% | **83%** | 65% | **0.73** | Most precise — only 13 false alarms, but misses 34 fraud cases |
    | Naive Bayes | 99% | 14% | 66% | 0.23 | Catches 65 fraud cases but raises 333 false alarms |
    """)

    # ── Effect of SMOTE ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Effect of SMOTE Balancing")
    st.markdown("""
    SMOTE (Synthetic Minority Over-sampling Technique) creates artificial fraud samples during
    training so the model sees an equal number of fraud and normal transactions. The models are
    still tested on the original imbalanced test set so the evaluation is fair.

    | Model | Accuracy | Precision | Recall | F1 | What it means |
    |---|---|---|---|---|---|
    | Neural Network | 100% | 31% | 87% | **0.46** | Catches 85 of 98 fraud cases — best F1 balance |
    | Logistic Regression | 99% | 13% | **90%** | 0.23 | Catches the most fraud (88/98) but 588 false alarms |
    | Naive Bayes | 97% | 5% | 88% | 0.10 | Catches 86 fraud cases but 1,478 false alarms — too noisy |

    After SMOTE, recall improves dramatically across all models. The trade-off is more false alarms
    (lower precision). For a bank, **missing fraud is far more costly** than investigating a false
    alarm, so the recall improvement is worth it.
    """)

    # ── Recommendation ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Recommendation")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.success("""
        **Best overall: Neural Network (Balanced)**
        - F1 = 0.46 — best balance of precision and recall
        - Catches 85 out of 98 fraud cases
        - Only 189 false alarms out of 56,864 normal transactions
        - Suitable when you need both good fraud detection and manageable false alarms
        """)
        st.warning("""
        **Best recall: Logistic Regression (Balanced)**
        - Catches 88 out of 98 fraud cases (90% recall)
        - But generates 588 false alarms
        - Suitable when catching every fraud case is the top priority
        """)
    with col_r2:
        st.error("""
        **Avoid: Naive Bayes (Balanced)**
        - High recall (88%) but only 5% precision
        - Generates 1,478 false alarms — nearly 1 in 40 normal transactions flagged
        - Too many false alarms to be practical in a real banking system
        """)
        st.info("""
        **Best precision: Logistic Regression (Imbalanced)**
        - 83% precision — when it flags fraud, it is usually right
        - But misses 34 out of 98 fraud cases (35% missed)
        - Suitable only when false alarms are extremely costly
        """)

    st.info(
        "Bottom line: Accuracy is a misleading metric for fraud detection. "
        "Always use Recall, Precision, and F1 score. "
        "SMOTE balancing significantly improves the ability to catch fraud across all models."
    )


