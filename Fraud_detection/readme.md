# Fraud Detection Pipeline

Second project of my Decode Labs internship. This one was interesting because the dataset is extremely imbalanced. Only 0.39% transactions are fraud out of 555,719 total. So the real challenge wasn't building a model, it was making sure the model actually learns to catch fraud instead of ignoring it.

## The Problem

If a model just predicts "legitimate" for everything, it gets 99.6% accuracy but catches zero fraud. That's useless. This project is about avoiding that trap.

## What I Did

Cleaned the data first. Removed columns like cardholder name, address and card number since they don't help detect fraud patterns.

Used IQR to detect outliers in transaction amounts and capped them using Winsorization instead of deleting rows.

Applied SMOTE to handle the class imbalance. One important thing I learned is that SMOTE should always be applied after the train/test split, not before. Doing it before causes data leakage and your results become unreliable.

Trained two models, Logistic Regression and Random Forest, and compared them using Precision, Recall and ROC-AUC. Accuracy is completely ignored here because it's misleading on imbalanced data.

## Dataset

Credit Card Fraud Detection, 555,719 transactions, 0.39% fraud rate.

## Tools

Python, Pandas, Scikit-learn, Imbalanced-learn, Matplotlib, Google Colab

Decode Labs Data Science Internship, July 2026
