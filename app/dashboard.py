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
    eda_cols = st.columns(3)
    for col, (fname, cap) in zip(eda_cols, [
        ("transaction_amount_distribution.png", "Transaction Amount Distribution"),
        ("fraud_vs_amount_boxplot.png",          "Fraud vs Amount"),
        ("correlation_heatmap.png",              "Feature Correlation Heatmap"),
    ]):
        p = os.path.join(VIZ_DIR, fname)
        if os.path.exists(p):
            col.image(p, caption=cap, use_container_width=True)

    st.subheader("Time and Transaction Distributions")
    t_cols = st.columns(2)
    for col, (fname, cap) in zip(t_cols, [
        ("transaction_time_distribution.png", "Transaction Time Distribution"),
        ("pca_feature_distribution.png",      "PCA Feature Distribution"),
    ]):
        p = os.path.join(VIZ_DIR, fname)
        if os.path.exists(p):
            col.image(p, caption=cap, use_container_width=True)

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
    st.caption("Green rows = Balanced (SMOTE). Accuracy is misleading — focus on Recall and ROC-AUC.")

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
    st.write("Side-by-side comparison of all models on Imbalanced vs Balanced data.")

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
    st.write("Confusion matrices for all three models on both Imbalanced and Balanced datasets.")

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

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Impact of Class Imbalance")
        st.markdown("""
        - All imbalanced models show ~99–100% accuracy — this is **misleading**.
        - A model that predicts every transaction as normal gets 99.83% accuracy but catches **zero fraud**.
        - **Neural Network** (Imbalanced): Recall 83%, Precision 32%, F1 0.46
        - **Logistic Regression** (Imbalanced): Recall 65%, Precision 83%, F1 0.73
        - **Naive Bayes** (Imbalanced): Recall 66%, Precision 14%, F1 0.23
        """)

        st.subheader("Effect of SMOTE Balancing")
        st.markdown("""
        SMOTE creates synthetic fraud samples so the model sees equal fraud and normal during training.
        - **Neural Network** (Balanced): Recall 87%, Precision 31%, F1 0.46
        - **Logistic Regression** (Balanced): Recall 90%, Precision 13%, F1 0.23
        - **Naive Bayes** (Balanced): Recall 88%, Precision 5%, F1 0.10

        Trade-off: more fraud caught (higher recall) but more false alarms (lower precision).
        For fraud detection, **missing fraud is more costly than a false alarm**, so high recall is preferred.
        """)

    with col_b:
        st.subheader("Model Comparison — All Results (from notebooks)")
        st.markdown("""
        **Imbalanced dataset:**

        | Model | Accuracy | Precision | Recall | F1 |
        |---|---|---|---|---|
        | Neural Network | 100% | 32% | 83% | 0.46 |
        | Logistic Regression | 100% | 83% | 65% | 0.73 |
        | Naive Bayes | 99% | 14% | 66% | 0.23 |

        **Balanced (SMOTE) dataset:**

        | Model | Accuracy | Precision | Recall | F1 |
        |---|---|---|---|---|
        | Logistic Regression | 99% | 13% | **90%** | 0.23 |
        | Neural Network | 100% | 31% | 87% | **0.46** |
        | Naive Bayes | 97% | 5% | 88% | 0.10 |
        """)

        st.subheader("Recommendation")
        st.markdown("""
        - **Neural Network (Balanced)** — best overall F1 score (0.46), good balance of precision and recall.
        - **Logistic Regression (Balanced)** — highest recall (90%), catches the most fraud.
        - **Naive Bayes (Balanced)** — high recall (88%) but very low precision (5%), too many false alarms.
        - **Logistic Regression (Imbalanced)** — best precision (83%) when false alarms must be minimised.
        """)

    st.info(
        "Bottom line: Never rely on accuracy alone for imbalanced datasets. "
        "Always evaluate Recall, Precision, and F1 score for fraud detection tasks."
    )


