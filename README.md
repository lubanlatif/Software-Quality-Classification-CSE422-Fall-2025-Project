# Software-Quality-Classification-CSE422-Fall-2025-Project
🚀 CSE422 Project: Software Quality Classification & Clustering.

AI project applying machine learning in Python to classify software quality. Implemented EDA, preprocessing, Logistic Regression, Naive Bayes, Neural Networks, and K-Means clustering with full evaluation using ROC-AUC and confusion matrices.

- Semester: Fall 2025
- Language: Python
- Platform: Google Colab

📌 Overview
This repository contains my CSE422 (Artificial Intelligence) lab project, completed in Fall 2025, where I applied machine learning techniques to analyze and predict software quality labels using Python.

The project covers the entire ML pipeline — from exploratory data analysis (EDA) and data preprocessing to supervised classification and unsupervised clustering.
All experiments were conducted using Google Colab.


🎯 Objectives
- Analyze a real-world software quality dataset
- Perform EDA to uncover patterns and relationships
- Build and evaluate multiple classification models
- Treat the problem as unsupervised and apply K-Means clustering
- Compare models using standard evaluation metrics


📊 Dataset Information
//Target Feature: Quality Label
//Problem Type: Multi-class Classification
//Features: Numerical software metrics
//Preprocessing Applied:
//Label encoding
//Feature scaling (StandardScaler)
//Stratified train–test split


🔍 Exploratory Data Analysis (EDA)
Target class distribution analysis (imbalance check)
Correlation heatmap using Seaborn
Pairwise feature relationships
Statistical summary of features



🤖 Machine Learning Models Used
✅ Supervised Learning

Logistic Regression
Naive Bayes
Neural Network (MLPClassifier)



🔄 Unsupervised Learning
K-Means Clustering
Elbow method for optimal cluster selection
Cluster visualization
Cluster vs Quality Label comparison



📈 Model Evaluation Metrics
Each supervised model was evaluated using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
ROC–AUC Score (One-vs-Rest)
ROC Curves (Multi-class)


📊 Comparative bar charts and visualizations are included for clear performance analysis.


Libraries:
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn


▶️ How to Run (Google Colab)
Open Google Colab
Upload the notebook (.ipynb)
Upload the dataset (software_quality_dataset.csv)
Run cells sequentially from top to bottom
No additional setup required.

🧠 Key Findings
Neural Network achieved the best overall performance due to non-linear learning capability
Logistic Regression provided stable and interpretable results
Naive Bayes showed lower performance due to feature dependency
K-Means clustering revealed meaningful natural groupings aligned with quality labels

⚠️ Challenges Faced
Handling class imbalance
Interpreting multi-class ROC curves
Choosing optimal number of clusters
Ensuring fair model comparison
Built and evaluated multiple machine learning models to classify software quality labels using Python. Performed EDA, supervised learning (Logistic Regression, Naive Bayes, Neural Network), and unsupervised clustering (K-Means), with performance evaluation using ROC-AUC, confusion matrices, and precision-recall metrics.
