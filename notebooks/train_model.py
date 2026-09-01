"""
Trains a fraud classifier (LightGBM) with SHAP explainability,
tracked end-to-end via MLflow hosted on DagsHub.
Run from notebooks/ folder, after preprocess.py.

Setup:
1. Create a DagsHub repo (free): https://dagshub.com
2. pip install dagshub mlflow
3. Add to .env: DAGSHUB_REPO_OWNER=your_username, DAGSHUB_REPO_NAME=your_repo
"""

import os
import pandas as pd
import lightgbm as lgb
import shap
import joblib
import mlflow
import dagshub
from dotenv import load_dotenv
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score

load_dotenv()

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_TOKEN")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

dagshub.init(
    repo_owner=os.getenv("DAGSHUB_REPO_OWNER"),
    repo_name=os.getenv("DAGSHUB_REPO_NAME"),
    mlflow=True,
)

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").iloc[:, 0]
y_test = pd.read_csv("y_test.csv").iloc[:, 0]

params = {
    "n_estimators": 75,
    "max_depth": 3,
    "learning_rate": 0.05,
    "class_weight": "balanced",
    "random_state": 42,
}

with mlflow.start_run(run_name="lightgbm_fraud_classifier"):
    mlflow.log_params(params)

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    mlflow.log_metrics(metrics)

    print(classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"]))
    print(metrics)

    mlflow.lightgbm.log_model(model, artifact_path="fraud_model", registered_model_name="claim_sentinel_fraud_model")

    # SHAP explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    joblib.dump(model, "fraud_model.pkl")
    joblib.dump(explainer, "shap_explainer.pkl")
    mlflow.log_artifact("fraud_model.pkl")
    mlflow.log_artifact("shap_explainer.pkl")

    print("Saved fraud_model.pkl, shap_explainer.pkl, and logged run to DagsHub MLflow")
