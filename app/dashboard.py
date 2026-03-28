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

# ─── Verified metrics from notebooks ──────────────────────────────────────────
METRICS = {
    "Neural Network": {
        "Imbalanced": {"Accuracy": 1.00, "Precision": 0.11, "Recall": 0.00, "ROC-AUC": 0.50},
        "Balanced":   {"Accuracy": 1.00, "Precision": 0.63, "Recall": 0.85, "ROC-AUC": 0.97},
    },
    "Logistic Regression": {
        "Imbalanced": {"Accuracy": 1.00, "Precision": 0.83, "Recall": 0.65, "ROC-AUC": 0.97},
        "Balanced":   {"Accuracy": 0.99, "Precision": 0.13, "Recall": 0.90, "ROC-AUC": 0.97},
    },
    "Naive Bayes": {
        "Imbalanced": {"Accuracy": 0.99, "Precision": 0.14, "Recall": 0.66, "ROC-AUC": 0.97},
        "Balanced":   {"Accuracy": 0.99, "Precision": 0.05, "Recall": 0.88, "ROC-AUC": 0.96},
    },
}

CONFUSION_MATRICES = {
    "Neural Network": {
        "Imbalanced": np.array([[56862, 2],    [98, 0]]),
        "Balanced":   np.array([[56849, 15],   [15, 83]]),
    },
    "Logistic Regression": {
        "Imbalanced": np.array([[56827, 37],   [34, 64]]),
        "Balanced":   np.array([[56282, 582],  [10, 88]]),
    },
    "Naive Bayes": {
        "Imbalanced": np.array([[56531, 333],  [33, 65]]),
        "Balanced":   np.array([[55218, 1646], [12, 86]]),
    },
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
        "ROC Curves",
        "Neural Network Training",
        "Key Insights",
        "Threshold Tuning",
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
                "ROC-AUC": f"{m['ROC-AUC']:.2%}",
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

    | Model | Imbalanced Recall | Balanced Recall |
    |---|---|---|
    | Neural Network | 0% | 85% |
    | Logistic Regression | 65% | 90% |
    | Naive Bayes | 66% | 88% |

    > **Key finding:** Models trained on raw imbalanced data achieve ~99.8% accuracy
    > but catch very little fraud. SMOTE balancing dramatically improves recall.

    Use the **sidebar** to navigate between sections.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", "284,807")
    col2.metric("Fraud Cases",        "492 (0.17%)")
    col3.metric("Test Set Size",      "56,962")
    col4.metric("Fraud in Test Set",  "98")

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
        col.metric("ROC-AUC",           f"{m['ROC-AUC']:.2%}")

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

    for metric in ["Precision (Fraud)", "Recall (Fraud)", "ROC-AUC"]:
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
        st.metric("ROC-AUC",           f"{m['ROC-AUC']:.2%}")

    st.markdown("---")

    # ── All models side by side ───────────────────────────────────────────────
    for dtype in ["Imbalanced", "Balanced"]:
        st.subheader(f"All Models — {dtype}")
        cols = st.columns(3)
        for col, model in zip(cols, list(CONFUSION_MATRICES.keys())):
            cm_all = CONFUSION_MATRICES[model][dtype]
            m_all  = METRICS[model][dtype]
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm_all, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Non-Fraud", "Fraud"],
                        yticklabels=["Non-Fraud", "Fraud"], ax=ax,
                        annot_kws={"size": 10})
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("Actual", fontsize=9)
            ax.set_title(model, fontsize=10)
            ax.tick_params(labelsize=8)
            col.pyplot(fig); plt.close()
            col.caption(
                f"Precision: {m_all['Precision']:.2%}  |  "
                f"Recall: {m_all['Recall']:.2%}  |  "
                f"AUC: {m_all['ROC-AUC']:.2f}"
            )

    st.markdown("---")
    st.subheader("Neural Network — Saved Images")
    cm_cols = st.columns(2)
    for col, (fname, cap) in zip(cm_cols, [
        ("confusion_matrix_imbalanced.png", "Neural Network — Imbalanced"),
        ("confusion_matrix_balanced.png",   "Neural Network — Balanced (SMOTE)"),
    ]):
        p = os.path.join(VIZ_DIR, fname)
        if os.path.exists(p):
            col.image(p, caption=cap, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ROC Curves
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ROC Curves":
    st.header("ROC Curve Comparison — All Models")

    # ── Combined chart ────────────────────────────────────────────────────────
    st.subheader("All Models — Imbalanced vs Balanced")
    style_map = {
        ("Neural Network",      "Imbalanced"): ("navy",       "--"),
        ("Neural Network",      "Balanced"):   ("navy",       "-"),
        ("Logistic Regression", "Imbalanced"): ("darkorange", "--"),
        ("Logistic Regression", "Balanced"):   ("darkorange", "-"),
        ("Naive Bayes",         "Imbalanced"): ("green",      "--"),
        ("Naive Bayes",         "Balanced"):   ("green",      "-"),
    }
    fpr_base = np.linspace(0, 1, 200)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, dtype), (color, ls) in style_map.items():
        auc = METRICS[model][dtype]["ROC-AUC"]
        tpr = fpr_base ** ((1 - auc) / auc) if auc > 0.5 else fpr_base
        ax.plot(fpr_base, tpr, color=color, linestyle=ls,
                label=f"{model} {dtype} (AUC={auc:.2f})")
    ax.plot([0,1],[0,1], "k--", label="Random Classifier (AUC=0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison — All Models")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    st.pyplot(fig); plt.close()
    st.caption("Note: Curves are approximated from reported AUC values, not from raw predictions.")

    st.markdown("---")

    # ── Per-model ROC charts ──────────────────────────────────────────────────
    st.subheader("Per-Model ROC Curves")
    model_colors = {
        "Neural Network":      "navy",
        "Logistic Regression": "darkorange",
        "Naive Bayes":         "green",
    }
    roc_cols = st.columns(3)
    for col, model in zip(roc_cols, list(METRICS.keys())):
        fig, ax = plt.subplots(figsize=(4, 3.5))
        for dtype, ls in [("Imbalanced", "--"), ("Balanced", "-")]:
            auc = METRICS[model][dtype]["ROC-AUC"]
            tpr = fpr_base ** ((1 - auc) / auc) if auc > 0.5 else fpr_base
            ax.plot(fpr_base, tpr, color=model_colors[model], linestyle=ls,
                    label=f"{dtype} (AUC={auc:.2f})")
        ax.plot([0,1],[0,1], "k--", linewidth=0.8)
        ax.set_xlabel("FPR", fontsize=9); ax.set_ylabel("TPR", fontsize=9)
        ax.set_title(model, fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        col.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Neural Network — Saved ROC Images")
    saved_cols = st.columns(3)
    for col, (fname, cap) in zip(saved_cols, [
        ("roc_curve_imbalanced.png", "Neural Network — Imbalanced"),
        ("roc_curve_balanced.png",   "Neural Network — Balanced"),
        ("roc_curve_comparison.png", "Neural Network — Comparison"),
    ]):
        p = os.path.join(VIZ_DIR, fname)
        if os.path.exists(p):
            col.image(p, caption=cap, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Neural Network Training
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Neural Network Training":
    st.header("Neural Network Training Visualisations")
    st.write(
        "The following visualisations were generated during Neural Network training "
        "on both the imbalanced and SMOTE-balanced datasets."
    )

    train_cols = st.columns(3)
    for col, (fname, cap) in zip(train_cols, [
        ("loss_curve.png",                    "Training Loss Curve"),
        ("balanced_training_data_smote.png",  "Balanced Training Data (SMOTE)"),
        ("class_distribution_imbalanced.png", "Class Distribution — Imbalanced"),
    ]):
        p = os.path.join(VIZ_DIR, fname)
        if os.path.exists(p):
            col.image(p, caption=cap, use_container_width=True)

    pca_cols = st.columns(2)
    for col, (fname, cap) in zip(pca_cols, [
        ("pca_feature_distribution.png",   "PCA Feature Distribution"),
        ("precision_recall_threshold.png", "Precision and Recall vs Threshold"),
    ]):
        p = os.path.join(VIZ_DIR, fname)
        if os.path.exists(p):
            col.image(p, caption=cap, use_container_width=True)

    st.markdown("---")
    st.subheader("Neural Network Performance Summary")
    col_imb, col_bal = st.columns(2)
    for col, dtype in zip([col_imb, col_bal], ["Imbalanced", "Balanced"]):
        m = METRICS["Neural Network"][dtype]
        cm = CONFUSION_MATRICES["Neural Network"][dtype]
        col.subheader(dtype)
        col.metric("Precision (Fraud)", f"{m['Precision']:.2%}")
        col.metric("Recall (Fraud)",    f"{m['Recall']:.2%}")
        col.metric("ROC-AUC",           f"{m['ROC-AUC']:.2%}")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-Fraud", "Fraud"],
                    yticklabels=["Non-Fraud", "Fraud"], ax=ax,
                    annot_kws={"size": 10})
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.set_title(f"Neural Network — {dtype}", fontsize=10)
        col.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Key Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Key Insights":
    st.header("Key Insights")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Impact of Class Imbalance")
        st.markdown("""
        - All imbalanced models show ~99–100% accuracy — this is misleading.
        - Neural Network (Imbalanced): **0% Recall** — catches zero fraud.
        - Logistic Regression (Imbalanced): 65% Recall, best precision (83%).
        - Naive Bayes (Imbalanced): 66% Recall, low precision (14%).
        """)

        st.subheader("Effect of SMOTE Balancing")
        st.markdown("""
        - Recall improves dramatically across all models after SMOTE.
        - Trade-off: more false positives (lower precision).
        - For fraud detection, **high Recall is critical** — missing fraud is costly.
        """)

    with col_b:
        st.subheader("Model Comparison (Balanced)")
        st.markdown("""
        | Model | Recall | Precision | AUC |
        |---|---|---|---|
        | Logistic Regression | **90%** | 13% | 0.97 |
        | Naive Bayes | 88% | 5% | 0.96 |
        | Neural Network | 85% | **63%** | **0.97** |
        """)

        st.subheader("Recommendation")
        st.markdown("""
        - **Neural Network (Balanced)** — best precision-recall balance overall.
        - **Logistic Regression (Balanced)** — highest recall, interpretable baseline.
        - **Naive Bayes** — fast and simple but lowest precision.
        - Use the Threshold Tuning page to adjust the precision-recall trade-off.
        """)

    st.info(
        "Bottom line: Never rely on accuracy alone for imbalanced datasets. "
        "Always evaluate Recall, Precision, and ROC-AUC for fraud detection tasks."
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Threshold Tuning
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Threshold Tuning":
    st.header("Threshold Tuning")
    st.write("Select dataset preparation type to adjust the classification threshold:")

    setup = st.selectbox("Dataset Setup", ["Imbalanced", "Balanced"])

    st.write(
        "Adjust the classification threshold to explore the precision-recall trade-off. "
        "Lower threshold flags more fraud (higher recall, lower precision)."
    )

    threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5, 0.05)

    base_cm = CONFUSION_MATRICES["Neural Network"][setup]
    tn_b, fp_b, fn_b, tp_b = base_cm[0,0], base_cm[0,1], base_cm[1,0], base_cm[1,1]
    total_fraud  = tp_b + fn_b
    total_normal = tn_b + fp_b

    recall_approx = min(1.0, (tp_b / total_fraud)  * (0.5 / threshold) ** 0.5)
    fpr_approx    = min(1.0, (fp_b / total_normal) * (0.5 / threshold) ** 0.5)
    tp_approx  = int(recall_approx * total_fraud)
    fn_approx  = total_fraud - tp_approx
    fp_approx  = int(fpr_approx * total_normal)
    tn_approx  = total_normal - fp_approx
    prec_approx = tp_approx / (tp_approx + fp_approx) if (tp_approx + fp_approx) > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall (Fraud Caught)", f"{recall_approx:.1%}")
    c2.metric("Precision",             f"{prec_approx:.1%}")
    c3.metric("True Positives",        tp_approx)
    c4.metric("False Positives",       fp_approx)

    col_cm, col_chart = st.columns([1, 1])
    with col_cm:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm_approx = np.array([[tn_approx, fp_approx], [fn_approx, tp_approx]])
        sns.heatmap(cm_approx, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-Fraud", "Fraud"],
                    yticklabels=["Non-Fraud", "Fraud"], ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(f"Confusion Matrix — Neural Network ({setup}, threshold={threshold})")
        st.pyplot(fig); plt.close()

    with col_chart:
        thresholds = np.arange(0.1, 0.95, 0.05)
        recalls, precisions = [], []
        for t in thresholds:
            r = min(1.0, (tp_b / total_fraud)  * (0.5 / t) ** 0.5)
            f = min(1.0, (fp_b / total_normal) * (0.5 / t) ** 0.5)
            tp_t = int(r * total_fraud)
            fp_t = int(f * total_normal)
            p = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
            recalls.append(r); precisions.append(p)

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(thresholds, recalls,    label="Recall",    color="steelblue")
        ax2.plot(thresholds, precisions, label="Precision", color="tomato")
        ax2.axvline(threshold, color="gray", linestyle="--", label=f"Current ({threshold})")
        ax2.set_xlabel("Threshold")
        ax2.set_ylabel("Score")
        ax2.set_title(f"Precision and Recall vs Threshold ({setup})")
        ax2.legend(); ax2.grid(alpha=0.3)
        st.pyplot(fig2); plt.close()

    st.caption(
        "Note: Simulation is approximate — scaled from the Neural Network baseline confusion matrix."
    )
