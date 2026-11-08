
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="wide"
)

st.title("Dry Bean Classification")
st.write("An interactive demonstration of five machine-learning classification models.")
st.caption("Dataset: UCI Dry Bean | Multiclass classification")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

# Check model files
missing = [
    filename for filename in MODEL_FILES.values()
    if not (MODEL_DIR / filename).exists()
]

if missing:
    st.error("Missing model files: " + ", ".join(missing))
    st.stop()

# Upload test data
st.subheader("1. Upload test data")
st.write("Upload test_data.csv containing 16 features and the target column Class.")

uploaded = st.file_uploader(
    "Choose a test CSV file",
    type=["csv"]
)

if uploaded is None:
    st.info("Please upload test_data.csv.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if "Class" not in data.columns:
    st.error("Target column 'Class' is missing.")
    st.stop()

if data.empty:
    st.error("Uploaded CSV contains no rows.")
    st.stop()

st.write("Preview of uploaded data")
st.dataframe(data.head(10), use_container_width=True)

# Model selection
st.subheader("2. Select a classification model")

model_name = st.selectbox(
    "Choose model",
    list(MODEL_FILES.keys())
)

try:
    model = joblib.load(MODEL_DIR / MODEL_FILES[model_name])
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

# IMPORTANT FIX:
# Individual model files contain Pipeline objects,
# NOT dictionaries containing feature_columns.

if hasattr(model, "feature_names_in_"):
    feature_columns = list(model.feature_names_in_)
else:
    feature_columns = [
        col for col in data.columns if col != "Class"
    ]

missing_columns = [
    col for col in feature_columns if col not in data.columns
]

if missing_columns:
    st.error(
        "Missing feature columns: " + ", ".join(missing_columns)
    )
    st.stop()

if len(feature_columns) != 16:
    st.error(
        f"Expected 16 feature columns, but found {len(feature_columns)}. "
        "Please check your test_data.csv."
    )
    st.stop()

X_test = data[feature_columns].apply(
    pd.to_numeric, errors="coerce"
)

y_test = data["Class"].astype(str)

if X_test.isna().any().any():
    st.error("Feature columns contain missing or non-numeric values.")
    st.stop()

# Run evaluation
if st.button("Run evaluation", type="primary"):

    try:
        predictions = model.predict(X_test)
        predictions = pd.Series(predictions).astype(str).to_numpy()

        classes = getattr(model, "classes_", None)

        if classes is None and hasattr(model, "steps"):
            classes = model.steps[-1][1].classes_

        classes = [str(c) for c in classes]

        scores = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions,
                average="weighted", zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions,
                average="weighted", zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions,
                average="weighted", zero_division=0
            ),
            "MCC": matthews_corrcoef(y_test, predictions)
        }

        try:
            probabilities = model.predict_proba(X_test)

            scores["AUC (weighted OVR)"] = roc_auc_score(
                y_test,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="weighted"
            )
        except (ValueError, AttributeError):
            scores["AUC (weighted OVR)"] = None

        st.success(f"Evaluation completed: {model_name}")

        # Metrics
        st.subheader("3. Evaluation metrics")

        cols = st.columns(3)

        for col, (label, value) in zip(cols * 2, scores.items()):
            shown = (
                "Not defined"
                if value is None or pd.isna(value)
                else f"{value:.4f}"
            )
            col.metric(label, shown)

        # Confusion matrix
        st.subheader("4. Confusion matrix")

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=classes
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=classes
        )

        display.plot(
            ax=ax,
            xticks_rotation=45,
            colorbar=False
        )

        ax.set_title(f"Confusion Matrix – {model_name}")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # Classification report
        st.subheader("5. Classification report")

        report = classification_report(
            y_test,
            predictions,
            zero_division=0,
            output_dict=True
        )

        st.dataframe(
            pd.DataFrame(report).transpose().round(4),
            use_container_width=True
        )

        # Predictions
        output = data.copy()
        output["Predicted_Class"] = predictions

        st.subheader("6. Predictions")

        st.dataframe(
            output.head(25),
            use_container_width=True
        )

        st.download_button(
            "Download predictions as CSV",
            output.to_csv(index=False).encode("utf-8"),
            file_name="dry_bean_predictions.csv",
            mime="text/csv"
        )

    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        st.exception(exc)
