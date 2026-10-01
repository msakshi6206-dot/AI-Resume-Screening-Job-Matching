import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# PATHS
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "Resume",
    "Resume_processed.csv"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

# ==================================================
# LOAD DATASET
# ==================================================
print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

df = pd.read_csv(
    DATASET_PATH
)

# ==================================================
# CLEAN DATA
# ==================================================

df["Processed_Resume"] = (
    df["Processed_Resume"]
    .fillna("")
    .astype(str)
)

df["Category"] = (
    df["Category"]
    .fillna("")
    .astype(str)
)

df = df[
    df["Processed_Resume"].str.strip() != ""
].copy()

X = df["Processed_Resume"]

y = df["Category"]

# ==================================================
# SAME TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

# ==================================================
# LOAD V1
# ==================================================

v1_vectorizer_path = os.path.join(
    MODELS_DIR,
    "tfidf_vectorizer.pkl"
)

v1_model_path = os.path.join(
    MODELS_DIR,
    "resume_classifier.pkl"
)

v1_vectorizer = joblib.load(
    v1_vectorizer_path
)

v1_model = joblib.load(
    v1_model_path
)

# ==================================================
# LOAD V2
# ==================================================

v2_vectorizer_path = os.path.join(
    MODELS_DIR,
    "tfidf_vectorizer_v2.pkl"
)

v2_model_path = os.path.join(
    MODELS_DIR,
    "resume_classifier_v2.pkl"
)

v2_vectorizer = joblib.load(
    v2_vectorizer_path
)

v2_model = joblib.load(
    v2_model_path
)

# ==================================================
# V1 PREDICTION
# ==================================================

print("\n")
print("=" * 70)
print("V1 MODEL")
print("=" * 70)

X_test_v1 = v1_vectorizer.transform(
    X_test
)

v1_predictions = v1_model.predict(
    X_test_v1
)

v1_accuracy = accuracy_score(
    y_test,
    v1_predictions
)

print(
    f"V1 Accuracy: {v1_accuracy * 100:.2f}%"
)

# ==================================================
# V2 PREDICTION
# ==================================================

print("\n")
print("=" * 70)
print("V2 MODEL")
print("=" * 70)

X_test_v2 = v2_vectorizer.transform(
    X_test
)

v2_predictions = v2_model.predict(
    X_test_v2
)

v2_accuracy = accuracy_score(
    y_test,
    v2_predictions
)

print(
    f"V2 Accuracy: {v2_accuracy * 100:.2f}%"
)

# ==================================================
# IMPROVEMENT
# ==================================================

improvement = (
    v2_accuracy - v1_accuracy
) * 100

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    f"V1 Accuracy : {v1_accuracy * 100:.2f}%"
)

print(
    f"V2 Accuracy : {v2_accuracy * 100:.2f}%"
)

print(
    f"Improvement : {improvement:+.2f} percentage points"
)

# ==================================================
# V2 CLASSIFICATION REPORT
# ==================================================

print("\n")
print("=" * 70)
print("V2 CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        v2_predictions,
        zero_division=0
    )
)

# ==================================================
# CONFUSION MATRIX
# ==================================================
labels = sorted(
    y.unique()
)

cm = confusion_matrix(
    y_test,
    v2_predictions,
    labels=labels
)

print("\n")
print("=" * 70)
print("V2 CONFUSION MATRIX")
print("=" * 70)

print(
    "Rows = Actual Category"
)

print(
    "Columns = Predicted Category"
)

print("\nCategories:")

for index, label in enumerate(labels):

    print(
        index,
        "->",
        label
    )

print("\nMatrix:\n")

print(cm)

# ==================================================
# FINAL
# ==================================================
print("\n")
print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)