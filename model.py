import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import model



def train_default_model(df):

    # -----------------------
    # Define Features & Target
    # -----------------------

    X = df[
        [
            "credit_score",
            "loan_amount",
            "monthly_income",
            "interest_rate",
            "debt_to_income_ratio",
            "loan_term_months",
            "industry",
            "region",
        ]
    ]

    y = df["default_flag"]

    # -----------------------
    # Train-Test Split
    # -----------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # -----------------------
    # Preprocessing
    # -----------------------

    categorical_features = ["industry", "region"]
    numeric_features = [
        "credit_score",
        "loan_amount",
        "monthly_income",
        "interest_rate",
        "debt_to_income_ratio",
        "loan_term_months",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    # -----------------------
    # Model
    # -----------------------

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    # -----------------------
    # Train
    # -----------------------

    pipeline.fit(X_train, y_train)

    # -----------------------
    # Evaluate
    # -----------------------

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("ROC AUC:", roc_auc)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return pipeline

df = pd.read_csv("synthetic_loans_100k.csv")

pipeline = model.train_default_model(df)