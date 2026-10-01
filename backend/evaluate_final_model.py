import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report
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
# HEADER
# ==================================================

print("=" * 70)
print("FINAL V3 MODEL EVALUATION")
print("=" * 70)


# ==================================================
# LOAD DATASET
# ==================================================

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
# TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==================================================
# LOAD V3 MODEL
# ==================================================

vectorizer_path = os.path.join(
    MODELS_DIR,
    "tfidf_vectorizer_v3.pkl"
)

model_path = os.path.join(
    MODELS_DIR,
    "resume_classifier_v3.pkl"
)


vectorizer = joblib.load(
    vectorizer_path
)

model = joblib.load(
    model_path
)


# ==================================================
# TRANSFORM TEST DATA
# ==================================================

X_test_tfidf = vectorizer.transform(
    X_test
)


# ==================================================
# PREDICTION
# ==================================================

y_pred = model.predict(
    X_test_tfidf
)


# ==================================================
# ACCURACY
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n")
print("=" * 70)
print("FINAL MODEL RESULT")
print("=" * 70)

print(
    f"\nV3 Accuracy: {accuracy * 100:.2f}%"
)


# ==================================================
# CLASSIFICATION REPORT
# ==================================================

print("\n")
print("=" * 70)
print("V3 CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ==================================================
# SUMMARY
# ==================================================

correct = (
    y_test == y_pred
).sum()

incorrect = (
    y_test != y_pred
).sum()

error_rate = (
    incorrect / len(y_test)
) * 100


print("\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    "Total test samples :",
    len(y_test)
)

print(
    "Correct predictions:",
    correct
)

print(
    "Incorrect predictions:",
    incorrect
)

print(
    f"Accuracy           : {accuracy * 100:.2f}%"
)

print(
    f"Error rate         : {error_rate:.2f}%"
)

print("\n")
print("=" * 70)
print("FINAL MODEL CONFIGURATION")
print("=" * 70)

print("Algorithm          : Linear SVM")
print("TF-IDF             : Unigram + Bigram")
print("Maximum features   : 10,000")
print("Class weighting     : Balanced")
print("C value             : 2.0")
print("Training samples    : 1,984")
print("Testing samples     : 497")
print(f"Final accuracy      : {accuracy * 100:.2f}%")

print("\n")
print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)