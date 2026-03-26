import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    layout="wide"
)

st.title("Credit Card Fraud Detection Comparative Study")

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Dataset Overview", "Exploratory Data Analysis", "Model Comparison"]
)

# DATASET OVERVIEW PAGE
if page == "Dataset Overview":

    st.header("Dataset Overview")

    # Dataset Shape 
    col1, col2 = st.columns(2)

    with col1:
        rows, cols = df.shape
        st.subheader("Dataset Shape")
        st.metric(label="Rows × Columns", value=f"{rows} × {cols}")

    with col2:
        fraud_rate = df['Class'].mean() * 100
        st.subheader("Fraud Rate")
        st.metric(label="Fraud Percentage", value=f"{fraud_rate:.2f}%")

    # Class Distribution 
    st.subheader("Class Distribution")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(
        x='Class',
        hue='Class',
        data=df,
        palette={0: "green", 1: "red"},
        ax=ax
    )
    ax.set_title("Fraud (red) vs Non-Fraud (green)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Amount and Time Range 
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Transaction Amount Range")
        min_amount = df['Amount'].min()
        max_amount = df['Amount'].max()
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(["Min Amount", "Max Amount"], [min_amount, max_amount],
               color=["lightgreen", "darkgreen"])
        ax.set_ylabel("Amount")
        ax.set_title("Amount Range")
        st.pyplot(fig)

    with col4:
        st.subheader("Transaction Time Range")
        min_time = df['Time'].min()
        max_time = df['Time'].max()
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(["Start Time", "End Time"], [min_time, max_time],
               color=["lightgreen", "darkgreen"])
        ax.set_ylabel("Time(seconds)")
        ax.set_title("Time Range")
        st.pyplot(fig)

elif page == "Exploratory Data Analysis":

    st.header("Exploratory Data Analysis")

    # Create Hour feature
    df['Hour'] = (df['Time'] // 3600) % 24

    col1, col2 = st.columns(2)

    # Transaction Amount Distribution 
    with col1:
        st.subheader("Amount Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df['Amount'], bins=50,kde=True, ax=ax)
        ax.set_title("Transaction Amount Distribution")
        ax.set_xlabel("Amount")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    #  Transaction Time Distribution 
    with col2:
        st.subheader("Transaction Time Distibution")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df['Time'], bins=50, kde=True, ax=ax)
        ax.set_title("Transactions Time Distibution")
        ax.set_xlabel("Time(seconds)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    col1, col2 = st.columns(2)

    #  Fraud vs Non-Fraud Amount 
    with col1:
        st.subheader("Fraud vs Non-Fraud Amount")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(
        x='Class',
        y='Amount',
        data=df,
        palette={"0": "skyblue", "1": "red"},
        ax=ax
    )
        ax.set_title("Transaction Amount by Class")
        ax.set_xlabel("Class (0 = Normal, 1 = Fraud)")
        ax.set_ylabel("Amount")
        st.pyplot(fig)
        
    
    
    # Fraud Transactions by Hour 
    with col2:
         st.subheader("Fraud Transactions by Hour")
         fig, ax = plt.subplots(figsize=(6,4))
         fraud_by_hour = df[df['Class']==1]['Hour'].value_counts().sort_index()
         ax.bar(fraud_by_hour.index, fraud_by_hour.values, color="red")
         ax.set_title("Fraud Transactions Throughout the Day")
         ax.set_xlabel("Hour")
         ax.set_ylabel("Number of Fraud Cases")
         st.pyplot(fig)
   
   
elif page == "Model Comparison":

    st.header("Model Performance Comparison")

    st.write("Select dataset preparation type to compare model performance:")

    # Dataset setup selector
    setup = st.selectbox(
        "Dataset Setup",
        ["Baseline", "Balanced & Scaled"]
    )
   


    results_dict = {
        "Baseline": pd.DataFrame({
            "Model": ["Naive Bayes", "Logistic Regression", "Neural Network"],

             # Precision
            "Precision Class 0": [1.00, 1.00, 1.00],
            "Precision Class 1": [0.14, 0.83, 0.11],

             # Recall
            "Recall Class 0": [0.99, 1.00, 1.00],
            "Recall Class 1": [0.66, 0.65, 0.00],
    
             # F1 Score
            "F1 Class 0": [0.99, 1.00, 1.00],
            "F1 Class 1": [0.23, 0.73, 0.00]
        }),
        "Balanced": pd.DataFrame({
            "Model": ["Naive Bayes", "Logistic Regression", "Neural Network"],
             # Precision
            "Precision Class 0": [1.00, 1.00, 1.00],
            "Precision Class 1": [0.05, 0.13, 0.63],

            # Recall
            "Recall Class 0": [0.99, 0.99, 1.00],
            "Recall Class 1": [0.88, 0.90, 0.85],

            # F1 Score
            "F1 Class 0": [0.99, 0.99, 1.00],
            "F1 Class 1": [0.10, 0.23, 0.72]
        })
    }
   

    results = results_dict[setup]

    #  Performance Table 
    st.subheader(f"{setup} Dataset Performance Table")
    st.dataframe(results)

    # Bar Chart 
    st.subheader(f"{setup} Dataset Performance Metrics")
    fig, ax = plt.subplots(figsize=(8,4))
    results.set_index("Model").plot(
        kind="bar",
        ax=ax,
        color=["skyblue", "lightgreen", "lightcoral"]
    )
    ax.set_title(f"{setup} Dataset - Precision, Recall, F1 Score by Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
   

   

