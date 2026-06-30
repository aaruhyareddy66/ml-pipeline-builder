from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import io
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample datasets
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

SAMPLE_DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}

MODELS = {
    "random_forest": RandomForestClassifier,
    "svm": SVC,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
}

# Store last uploaded dataset in memory
session_data = {}

@app.get("/")
def root():
    return {"message": "ML Pipeline Builder API Running!"}

@app.get("/sample_datasets")
def get_sample_datasets():
    return {"datasets": list(SAMPLE_DATASETS.keys())}

@app.post("/load_sample")
def load_sample(dataset_name: str = Form(...)):
    if dataset_name not in SAMPLE_DATASETS:
        return {"error": "Dataset not found"}
    data = SAMPLE_DATASETS[dataset_name]()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    session_data["df"] = df
    return {
        "columns": list(df.columns),
        "rows": len(df),
        "preview": df.head(5).to_dict(orient="records")
    }

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        session_data["df"] = df
        return {
            "columns": list(df.columns),
            "rows": len(df),
            "preview": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/run_pipeline")
def run_pipeline(
    target_column: str = Form(...),
    preprocessing: str = Form(...),  # "scale", "none"
    handle_missing: str = Form(...),  # "drop", "mean", "none"
    model_name: str = Form(...),
    test_size: float = Form(0.2)
):
    try:
        if "df" not in session_data:
            return {"error": "No dataset loaded. Upload or load a sample first."}

        df = session_data["df"].copy()

        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found"}

        # Handle missing values
        if handle_missing == "drop":
            df = df.dropna()
        elif handle_missing == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            imputer = SimpleImputer(strategy="mean")
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categorical columns
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y.astype(str))

        # Scale
        if preprocessing == "scale":
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Train
        if model_name not in MODELS:
            return {"error": "Model not found"}

        model_class = MODELS[model_name]
        if model_name == "svm":
            model = model_class(probability=True)
        else:
            model = model_class(random_state=42) if model_name != "logistic_regression" else model_class(max_iter=1000)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds).tolist()
        report = classification_report(y_test, preds, output_dict=True)

        return {
            "accuracy": round(acc * 100, 2),
            "confusion_matrix": cm,
            "classification_report": report,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "features_used": list(X.columns),
            "model_used": model_name
        }
    except Exception as e:
        return {"error": str(e)}