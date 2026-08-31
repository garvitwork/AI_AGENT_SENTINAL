"""
Trains a fraud classifier (LightGBM) with SHAP explainability.
Run from notebooks/ folder, after preprocess.py.
"""

import pandas as pd
import lightgbm as lgb
import shap
import joblib
from sklearn.metrics import classification_report, roc_auc_score

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").iloc[:, 0]
y_test = pd.read_csv("y_test.csv").iloc[:, 0]

model = lgb.LGBMClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.05,
    class_weight="balanced",   # handles the 75/25 imbalance
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# SHAP explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

joblib.dump(model, "fraud_model.pkl")
joblib.dump(explainer, "shap_explainer.pkl")
print("Saved fraud_model.pkl and shap_explainer.pkl")