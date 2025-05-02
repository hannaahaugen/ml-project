# below is a very simple model, I just did a random sample of 20% as test set 

# update for now data with missing column has just been removed

# bro its so ass rn, btw default threshold is 0.5

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import statsmodels.api as sm  # For OLS summary

# Load data
data_folder = Path("data")
file_path = data_folder / "features_non_zero_2639.xlsx"
df = pd.read_excel(file_path)

# Prepare features (X) and target (y)
X = df.iloc[:, 1:-1]  # All columns except first (date) and last (target)
y = df.iloc[:, -1]    # Last column (binary target)

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- OLS Regression (for detailed coefficients) ---
# Add constant for intercept term
X_train_ols = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_ols).fit()

# Print OLS summary
print("\n=== OLS Regression Summary ===")
print(ols_model.summary())  # Shows coefficients, R², t-values, p-values

# --- Logistic Regression (for classification metrics) ---
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Classification metrics
print("\n=== Classification Metrics ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Data info
print("\n=== Data Info ===")
print(f"Total samples: {len(df)}")
print(f"Positive class ratio: {y.mean():.4f}")