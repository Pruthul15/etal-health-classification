# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For saving models
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV  # Added GridSearchCV for tuning
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve

# Step 2: Load the dataset
file_path = "fetalhealth-1-y0mhk0cb (1).csv"
df = pd.read_csv(file_path)

# Step 3: Data Preprocessing
X = df.drop(columns=['fetal_health'])  # Features
y = df['fetal_health']  # Target

# Splitting the dataset before SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Applying SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 4: Scale the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Step 5: Hyperparameter Tuning with GridSearchCV
# Define parameter grids
log_reg_params = {'C': [0.01, 0.1, 1, 10], 'max_iter': [2000, 5000], 'solver': ['saga']}
rf_params = {'n_estimators': [100, 150, 200], 'max_depth': [None, 10, 20]}

# Perform GridSearchCV
log_reg_tuned = GridSearchCV(LogisticRegression(random_state=42), log_reg_params, cv=3, scoring='accuracy', n_jobs=-1)
log_reg_tuned.fit(X_train_scaled, y_train_resampled)

rf_clf_tuned = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy', n_jobs=-1)
rf_clf_tuned.fit(X_train_resampled, y_train_resampled)

# Get best models
best_log_reg = log_reg_tuned.best_estimator_
best_rf_clf = rf_clf_tuned.best_estimator_

# Step 6: Predictions
y_pred_log_reg = best_log_reg.predict(X_test_scaled)
y_pred_rf = best_rf_clf.predict(X_test)

# Step 7: Evaluate Both Models
log_reg_acc = accuracy_score(y_test, y_pred_log_reg)
rf_acc = accuracy_score(y_test, y_pred_rf)

log_reg_roc_auc = roc_auc_score(y_test, best_log_reg.predict_proba(X_test_scaled), multi_class='ovr')
rf_roc_auc = roc_auc_score(y_test, best_rf_clf.predict_proba(X_test), multi_class='ovr')

log_reg_report = classification_report(y_test, y_pred_log_reg)
rf_report = classification_report(y_test, y_pred_rf)

# Print Performance Metrics
print("\n🔹 Logistic Regression Performance (Tuned):")
print(f"Best Parameters: {log_reg_tuned.best_params_}")
print(f"Accuracy: {log_reg_acc:.4f}")
print(f"ROC AUC Score: {log_reg_roc_auc:.4f}")
print("Classification Report:\n", log_reg_report)

print("\n🔹 Random Forest Performance (Tuned):")
print(f"Best Parameters: {rf_clf_tuned.best_params_}")
print(f"Accuracy: {rf_acc:.4f}")
print(f"ROC AUC Score: {rf_roc_auc:.4f}")
print("Classification Report:\n", rf_report)

# Step 8: Visualizing Confusion Matrices
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_log_reg), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# Step 9: Feature Importance Analysis
feature_importances = pd.Series(best_rf_clf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Step 10: Save the trained models
joblib.dump(best_log_reg, "logistic_regression_model.pkl")
joblib.dump(best_rf_clf, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Models and Scaler Saved Successfully!")

# Step 11: ROC Curve Visualization (Automatically selecting classes)
plt.figure(figsize=(10, 5))

for model, name, X_test_scaled_data in zip([best_log_reg, best_rf_clf], ['Logistic Regression', 'Random Forest'], [X_test_scaled, X_test]):
    y_score = model.predict_proba(X_test_scaled_data)
    
    for class_label in range(1, 4):  # Assuming classes are 1, 2, 3
        fpr, tpr, _ = roc_curve(y_test, y_score[:, class_label - 1], pos_label=class_label)
        plt.plot(fpr, tpr, label=f"{name} (Class {class_label}, AUC = {auc(fpr, tpr):.2f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (All Classes)")
plt.legend()
plt.show()

# Step 12: Precision-Recall Curve
plt.figure(figsize=(10, 5))

for model, name, X_test_scaled_data in zip([best_log_reg, best_rf_clf], ['Logistic Regression', 'Random Forest'], [X_test_scaled, X_test]):
    y_score = model.predict_proba(X_test_scaled_data)
    
    for class_label in range(1, 4):  # Assuming classes are 1, 2, 3
        precision, recall, _ = precision_recall_curve(y_test, y_score[:, class_label - 1], pos_label=class_label)
        plt.plot(recall, precision, label=f"{name} (Class {class_label})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison (All Classes)")
plt.legend()
plt.show()
