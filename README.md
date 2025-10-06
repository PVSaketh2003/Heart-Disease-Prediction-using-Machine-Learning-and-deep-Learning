# Heart-Disease-Prediction-using-Machine-Learning-and-deep-Learning

🩺 Heart Disease Prediction using Machine Learning & Deep Learning

This project focuses on predicting the likelihood of heart disease in patients using a variety of machine learning and deep learning techniques.
It uses a structured dataset containing patient health attributes such as age, cholesterol, resting blood pressure, thalassemia, and more.

🔍 Project Overview

Dataset Size: 1025 records, 14 features

Goal: Classify whether a person is likely to have heart disease (1) or not (0)

Tech Stack: Python, Pandas, NumPy, Scikit-learn, Keras, Matplotlib, Seaborn

⚙️ Key Steps

Data Preprocessing:

Handled missing values, renamed columns, and removed outliers.

Scaled features using StandardScaler.

Exploratory Data Analysis (EDA):

Visualized correlations between features like chest pain type, thalassemia, and heart disease risk using Seaborn.

Model Training:

Built and compared multiple classification models:

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Naive Bayes

Tuned hyperparameters with RandomizedSearchCV for optimal performance.

Deep Learning Model:

Designed a Neural Network (ANN) with multiple dense layers using Keras.

Achieved strong accuracy on validation data (≈95–97%).

🧠 Best Performing Models

KNN (Tuned): 100% test accuracy

Random Forest & Decision Tree: ~98.5% test accuracy

Neural Network: ~95.7% validation accuracy

📊 Evaluation Metrics

Accuracy

Precision, Recall, F1-Score

Confusion Matrix visualization

📁 How to Use

Download or clone the repository.

Run the Jupyter Notebook or Python file provided.

Install dependencies using pip install -r requirements.txt.

The trained model will display predictions and performance metrics.

🎯 Objective:
This project demonstrates how multiple supervised learning algorithms and deep learning architectures can be applied and optimized for a medical classification problem, emphasizing interpretability, model comparison, and performance tuning.
