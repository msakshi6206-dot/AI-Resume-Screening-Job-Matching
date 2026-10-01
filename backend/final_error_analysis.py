import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
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

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# ==================================================
# HEADER
# ==================================================

print("=" * 70)
print("FINAL V3 ERROR ANALYSIS")
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
# SAME TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

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
# PREDICTION
# ==================================================

X_test_tfidf = vectorizer.transform(
    X_test
)

y_pred = model.predict(
    X_test_tfidf
)

# ==================================================
# CREATE RESULTS DATAFRAME
# ==================================================

results = pd.DataFrame({

    "Actual_Category":
        y_test.values,

    "Predicted_Category":
        y_pred,

    "Resume_Text":
        X_test.values

})

# ==================================================
# WRONG PREDICTIONS
# ==================================================

errors = results[
    results["Actual_Category"]
    !=
    results["Predicted_Category"]
].copy()

# ==================================================
# ERROR SUMMARY
# ==================================================

print("\n")
print("=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

print(
    "Total test resumes:",
    len(results)
)

print(
    "Correct predictions:",
    len(results) - len(errors)
)

print(
    "Incorrect predictions:",
    len(errors)
)

error_rate = (
    len(errors)
    /
    len(results)
) * 100

print(
    f"Error rate: {error_rate:.2f}%"
)

# ==================================================
# MOST COMMON CONFUSIONS
# ==================================================

confusions = (
    errors
    .groupby(
        [
            "Actual_Category",
            "Predicted_Category"
        ]
    )
    .size()
    .reset_index(
        name="Count"
    )
    .sort_values(
        "Count",
        ascending=False
    )
)

print("\n")
print("=" * 70)
print("MOST COMMON V3 CATEGORY CONFUSIONS")
print("=" * 70)

print(
    confusions.head(20).to_string(
        index=False
    )
)

# ==================================================
# ERRORS BY CATEGORY
# ==================================================

category_errors = (
    errors
    .groupby(
        "Actual_Category"
    )
    .size()
    .reset_index(
        name="Error_Count"
    )
    .sort_values(
        "Error_Count",
        ascending=False
    )
)

print("\n")
print("=" * 70)
print("V3 ERRORS BY ACTUAL CATEGORY")
print("=" * 70)

print(
    category_errors.to_string(
        index=False
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
    y_pred,
    labels=labels
)

fig, ax = plt.subplots(
    figsize=(16, 14)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=True
)

plt.title(
    "Confusion Matrix - Final V3 Resume Classification Model",
    fontsize=16
)

plt.xlabel(
    "Predicted Category"
)

plt.ylabel(
    "Actual Category"
)

plt.tight_layout()

# ==================================================
# SAVE CONFUSION MATRIX
# ==================================================

matrix_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix_v3.png"
)

plt.savefig(
    matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==================================================
# SAVE ERROR FILES
# ==================================================

errors_path = os.path.join(
    RESULTS_DIR,
    "classification_errors_v3.csv"
)

errors.to_csv(
    errors_path,
    index=False
)

confusions_path = os.path.join(
    RESULTS_DIR,
    "category_confusions_v3.csv"
)

confusions.to_csv(
    confusions_path,
    index=False
)


category_errors_path = os.path.join(
    RESULTS_DIR,
    "category_error_summary_v3.csv"
)

category_errors.to_csv(
    category_errors_path,
    index=False
)

# ==================================================
# FINAL OUTPUT
# ==================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    "V3 Confusion Matrix:"
)

print(
    matrix_path
)

print(
    "\nV3 Classification Errors:"
)

print(
    errors_path
)

print(
    "\nV3 Category Confusions:"
)

print(
    confusions_path
)

print(
    "\nV3 Category Error Summary:"
)

print(
    category_errors_path
)
print("\n")
print("=" * 70)
print("FINAL V3 ERROR ANALYSIS COMPLETED")
print("=" * 70)