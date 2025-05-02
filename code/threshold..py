# experiment with threshold, 0.1 t0 0.9, increments of 0.1

# F1 peaks at 0.4 but is still super ass

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix)

# Load data
data_folder = Path("data")
file_path = data_folder / "features_non_zero_2639.xlsx"
df = pd.read_excel(file_path)

# Prepare features and target
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                   random_state=42, stratify=y)

# Train model with class weights
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train, y_train)
y_prob = logreg.predict_proba(X_test)[:, 1]  # Get probabilities for class 1

# Experiment with thresholds
thresholds = np.arange(0.1, 1.0, 0.1)  # 0.1 to 0.9 in 0.1 increments
results = []

for thresh in thresholds:
    y_pred = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    results.append({
        'Threshold': thresh,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'TP': cm[1, 1],  # True Positives
        'FP': cm[0, 1],   # False Positives
        'TN': cm[0, 0],   # True Negatives
        'FN': cm[1, 0]    # False Negatives
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Enhanced Plotting
plt.figure(figsize=(15, 6))

# Plot 1: Precision-Recall Tradeoff
plt.subplot(1, 3, 1)
plt.plot(results_df['Recall'], results_df['Precision'], 'b-o')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)

# Plot 2: Accuracy and F1 Score
plt.subplot(1, 3, 2)
plt.plot(results_df['Threshold'], results_df['Accuracy'], 'g-o', label='Accuracy')
plt.plot(results_df['Threshold'], results_df['F1'], 'r-o', label='F1 Score')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Accuracy & F1 Score vs Threshold')
plt.legend()
plt.grid(True)

# Plot 3: Confusion Matrix at Optimal F1 Threshold
optimal_idx = results_df['F1'].idxmax()
optimal_thresh = results_df.loc[optimal_idx, 'Threshold']
y_pred_optimal = (y_prob >= optimal_thresh).astype(int)
cm = confusion_matrix(y_test, y_pred_optimal)

plt.subplot(1, 3, 3)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix (Threshold={optimal_thresh:.1f})\nF1={results_df.loc[optimal_idx, "F1"]:.2f}')
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ['Predicted 0', 'Predicted 1'])
plt.yticks(tick_marks, ['Actual 0', 'Actual 1'])
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Add text annotations
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

# Print detailed results
print("\n=== Detailed Performance Across Thresholds ===")
print(results_df.round(4))

print("\n=== Optimal Threshold ===")
print(f"Best F1 Score: {results_df['F1'].max():.4f} at threshold: {optimal_thresh:.1f}")
print(f"At this threshold:")
print(f"- Precision: {results_df.loc[optimal_idx, 'Precision']:.4f}")
print(f"- Recall: {results_df.loc[optimal_idx, 'Recall']:.4f}")
print(f"- Accuracy: {results_df.loc[optimal_idx, 'Accuracy']:.4f}")
print("\nConfusion Matrix at Optimal Threshold:")
print(confusion_matrix(y_test, y_pred_optimal))