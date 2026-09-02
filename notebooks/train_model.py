"""
Trains a fraud classifier (LightGBM) with SHAP explainability,
tracked via MLflow on DagsHub. Reads hyperparams from params.yaml.
DVC stage: run from project root as `python notebooks/train_model.py`.
"""

import os
import json
import yaml
import pandas as pd
import lightgbm as lgb
import shap
import joblib
import mlflow
import dagshub
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
)

load_dotenv()

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_TOKEN")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

dagshub.init(
    repo_owner=os.getenv("DAGSHUB_REPO_OWNER"),
    repo_name=os.getenv("DAGSHUB_REPO_NAME"),
    mlflow=True,
)

with open("params.yaml") as f:
    all_params = yaml.safe_load(f)
rc_params = all_params["risk_classifier"]

X_train = pd.read_csv("notebooks/X_train.csv")
X_test = pd.read_csv("notebooks/X_test.csv")
y_train = pd.read_csv("notebooks/y_train.csv").iloc[:, 0]
y_test = pd.read_csv("notebooks/y_test.csv").iloc[:, 0]

params = {
    "n_estimators": rc_params["n_estimators"],
    "max_depth": rc_params["max_depth"],
    "learning_rate": rc_params["learning_rate"],
    "class_weight": "balanced",
    "random_state": 42,
}

os.makedirs("outputs", exist_ok=True)

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

    with open("outputs/risk_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Fraud", "Fraud"])
    disp.plot(cmap="Blues")
    plt.title("Fraud Classifier Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
    plt.close()

    mlflow.lightgbm.log_model(model, artifact_path="fraud_model", registered_model_name="claim_sentinel_fraud_model")

    # SHAP explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    joblib.dump(model, "notebooks/fraud_model.pkl")
    joblib.dump(explainer, "notebooks/shap_explainer.pkl")
    mlflow.log_artifact("notebooks/fraud_model.pkl")
    mlflow.log_artifact("notebooks/shap_explainer.pkl")
    mlflow.log_artifact("outputs/confusion_matrix.png")

    print("Saved fraud_model.pkl, shap_explainer.pkl, risk_metrics.json, confusion_matrix.png")
