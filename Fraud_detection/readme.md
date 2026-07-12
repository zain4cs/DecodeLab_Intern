# Fraud Detection Pipeline

This is my second project from the Decode Labs Data Science internship. Honestly, this one was more challenging than the first because the problem itself is tricky — you're trying to find fraud in a dataset where barely 0.4% of transactions are actually fraudulent. Everything else looks normal.

## What's the problem?

Imagine you build a model and it says "everything is legitimate" for every single transaction. Guess what — it'll be 99.6% accurate. Sounds great, right? But it caught zero fraud. That's exactly the trap this project is about avoiding.

The dataset has 555,719 real credit card transactions. Out of those, only 2,145 are fraud. So the model has to be smart enough to find that tiny minority without just ignoring them.

## What I did

**First things first — cleaned the data.**
Dropped columns that had no business being in a machine learning model. Things like the cardholder's name, home address, card number — these don't help the model learn fraud patterns, they just add noise.

**Handled outliers.**
Transaction amounts had some extreme values. Used IQR + Winsorization to cap them instead of deleting rows, because every row matters when your fraud class is already tiny.

**The big one — SMOTE.**
Since fraud cases were so rare, the model would just learn to ignore them. SMOTE generates synthetic fraud samples so the model actually gets to learn what fraud looks like. Important thing I learned here — you have to apply SMOTE after splitting the data, not before. If you do it before, your test set gets contaminated and your results are fake.

**Trained two models and compared them.**
- Logistic Regression — simple and fast, good baseline
- Random Forest — more powerful, handles complex patterns better

**Evaluated properly.**
Forgot about accuracy completely. Used Precision, Recall, and ROC-AUC instead. For fraud detection, missing a fraud (False Negative) is way more costly than a false alarm.

## Dataset

Credit Card Fraud Detection — 555,719 transactions, 23 columns, 0.39% fraud rate.

## Tools used

Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Google Colab

## What I learned

Honestly the biggest takeaway was that high accuracy means nothing in imbalanced problems. A model that scores 99% can still be completely useless. ROC-AUC and Recall are what actually matter when you're building something for real-world fraud detection.

---

*Decode Labs Data Science Internship — July 2026*
