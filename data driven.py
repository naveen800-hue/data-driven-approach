# A Data-Driven Approach for Customer Lifetime Value (CLV) in E-Commerce

# Step 1: Importing Required Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

# Step 2: Load Dataset
dataset_path = os.path.join('Datasets', 'Dataset.csv')
df = pd.read_csv(dataset_path)

# Step 3: Visual Exploration
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show()

# Count plot for Target classes
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='Target', data=df)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.title('Count Plot for Target Variable')
plt.show()

# Step 4: Preprocessing - Label Encoding
encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Step 5: Splitting Features and Target
target_column = 'Target'
X = df.drop(columns=[target_column])
y = df[target_column]
labels = sorted(y.unique())

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

# Step 7: Metric Calculation Function
precision = []
recall = []
fscore = []
accuracy = []

def calculateMetrics(algorithm, testY, predict):
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    print(f"{algorithm} Accuracy    : {a:.2f}")
    print(f"{algorithm} Precision   : {p:.2f}")
    print(f"{algorithm} Recall      : {r:.2f}")
    print(f"{algorithm} F1 Score     : {f:.2f}")

    print(f"\n{algorithm} Classification Report\n")
    print(classification_report(testY, predict, target_names=[str(l) for l in labels]))

    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="Blues", fmt="g")
    ax.set_ylim([0, len(labels)])
    plt.title(f"{algorithm} Confusion Matrix")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()

# Step 8: Logistic Regression Classifier (replaces invalid Lasso)
log_model_path = os.path.join('model', 'LogisticRegression.pkl')
if os.path.exists(log_model_path):
    LC = joblib.load(log_model_path)
    print("Logistic Regression model loaded successfully.")
else:
    LC = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
    LC.fit(X_train, y_train)
    os.makedirs('model', exist_ok=True)
    joblib.dump(LC, log_model_path)
    print("Logistic Regression model trained and saved successfully.")

predict_log = LC.predict(X_test)
calculateMetrics("Logistic Regression Classifier", y_test, predict_log)

# Step 9: Decision Tree Classifier
dt_model_path = os.path.join('model', 'DTC.pkl')
if os.path.exists(dt_model_path):
    DTC = joblib.load(dt_model_path)
    print("Decision Tree model loaded successfully.")
else:
    DTC = DecisionTreeClassifier()
    DTC.fit(X_train, y_train)
    joblib.dump(DTC, dt_model_path)
    print("Decision Tree model trained and saved successfully.")

predict_dtc = DTC.predict(X_test)
calculateMetrics("Decision Tree Classifier", y_test, predict_dtc)

# Step 10: Predicting on New Test Data
test_path = os.path.join('Datasets', 'testdata.csv')
test = pd.read_csv(test_path)

# Label encoding test data using same encoders
for col in test.columns:
    if col in encoders:
        test[col] = encoders[col].transform(test[col])
    elif test[col].dtype == 'object':
        # Fallback for unseen columns: fit fresh encoder (risky)
        le = LabelEncoder()
        test[col] = le.fit_transform(test[col])

# Predict using DTC
predictions = DTC.predict(test)
test['Prediction'] = predictions

# Display final test predictions
print("\nTest Data Predictions:")
print(test)
