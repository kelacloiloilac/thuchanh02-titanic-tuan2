"""Xây dựng mô hình dự đoán sống sót trên bộ dữ liệu Titanic.

Chạy từ thư mục dự án:
    python titanic/titanic_model.py

Hoặc chạy trực tiếp từ thư mục titanic:
    python titanic_model.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RANDOM_STATE = 42


def find_data_dir() -> Path:
    """Tìm thư mục chứa train.csv và test.csv từ vị trí chạy script."""
    candidates = [Path(__file__).resolve().parent, Path.cwd() / "titanic", Path.cwd()]
    for directory in candidates:
        if (directory / "train.csv").exists() and (directory / "test.csv").exists():
            return directory
    raise FileNotFoundError("Không tìm thấy train.csv và test.csv trong thư mục titanic.")


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng có ý nghĩa từ dữ liệu gốc."""
    result = data.copy()
    result["Title"] = (
        result["Name"]
        .str.extract(r",\s*([^.]*)\.", expand=False)
        .fillna("Unknown")
        .str.strip()
        .replace(
            {
                "Mlle": "Miss",
                "Ms": "Miss",
                "Mme": "Mrs",
            }
        )
    )
    common_titles = {"Mr", "Miss", "Mrs", "Master"}
    result.loc[~result["Title"].isin(common_titles), "Title"] = "Rare"
    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
    result["IsAlone"] = (result["FamilySize"] == 1).astype(int)
    result["IsChild"] = (result["Age"] < 16).astype(int)
    result["IsMother"] = (
        (result["Sex"] == "female")
        & (result["Parch"] > 0)
        & (result["Age"] > 18)
        & (result["Title"] == "Mrs")
    ).astype(int)
    result["HasCabin"] = result["Cabin"].notna().astype(int)
    result["Deck"] = result["Cabin"].str[0].fillna("U")
    result["Fare_Pclass"] = result["Fare"] / result["Pclass"]
    result["Age_Pclass"] = result["Age"] * result["Pclass"]
    result["TicketPrefix"] = (
        result["Ticket"]
        .astype(str)
        .str.replace(r"\d", "", regex=True)
        .str.replace(r"[\s./]+", "", regex=True)
        .replace("", "NONE")
    )
    return result


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "Age",
        "Fare",
        "FamilySize",
        "Fare_Pclass",
        "Age_Pclass",
        "SibSp",
        "Parch",
    ]
    categorical_features = [
        "Pclass",
        "Sex",
        "Embarked",
        "Title",
        "IsAlone",
        "IsChild",
        "IsMother",
        "Deck",
        "HasCabin",
        "TicketPrefix",
    ]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }
    return {
        name: Pipeline([("preprocessor", build_preprocessor()), ("model", model)])
        for name, model in models.items()
    }


def main() -> None:
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    engineered_train = add_features(train)
    engineered_test = add_features(test)
    feature_columns = [
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked",
        "Title",
        "FamilySize",
        "IsAlone",
        "IsChild",
        "IsMother",
        "Deck",
        "HasCabin",
        "Fare_Pclass",
        "Age_Pclass",
        "TicketPrefix",
        "SibSp",
        "Parch",
    ]
    X = engineered_train[feature_columns]
    y = train["Survived"]
    X_test = engineered_test[feature_columns]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    results = []
    fitted_models = {}

    print(f"Dữ liệu: train={train.shape}, test={test.shape}")
    print("\nTỷ lệ giá trị thiếu trong train:")
    print(train.isna().sum()[train.isna().sum() > 0].sort_values(ascending=False))

    for name, model in build_models().items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        model.fit(X_train, y_train)
        validation_probabilities = model.predict_proba(X_valid)[:, 1]
        validation_predictions = (validation_probabilities >= 0.5).astype(int)
        results.append(
            {
                "Model": name,
                "CV_Accuracy": scores["test_accuracy"].mean(),
                "CV_F1": scores["test_f1"].mean(),
                "CV_ROC_AUC": scores["test_roc_auc"].mean(),
                "Validation_Accuracy": accuracy_score(y_valid, validation_predictions),
                "Validation_F1": f1_score(y_valid, validation_predictions),
                "Validation_ROC_AUC": roc_auc_score(y_valid, validation_probabilities),
            }
        )
        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values(
        ["CV_ROC_AUC", "CV_Accuracy"], ascending=False
    )
    best_name = results_df.iloc[0]["Model"]
    validation_model = fitted_models[best_name]
    validation_predictions = validation_model.predict(X_valid)
    best_model = validation_model
    best_model.fit(X, y)

    print("\nKết quả mô hình:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nMô hình được chọn: {best_name}")
    print("\nBáo cáo validation:")
    print(classification_report(y_valid, validation_predictions))

    predictions = (best_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    submission = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Survived": predictions}
    )
    submission.to_csv(data_dir / "submission.csv", index=False)
    results_df.to_csv(data_dir / "model_comparison.csv", index=False)
    joblib.dump(best_model, data_dir / "titanic_model.joblib")
    with (data_dir / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "selected_model": best_name,
                "random_state": RANDOM_STATE,
                "features": feature_columns,
                "train_rows": len(train),
                "test_rows": len(test),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nĐã tạo: {data_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
