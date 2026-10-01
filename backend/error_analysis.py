import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


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
# LOAD DATASET
# ==================================================

print("=" * 70)
print("RESUME CLASSIFICATION ERROR ANALYSIS")
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
# TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ==================================================
# LOAD V2 MODEL
# ==================================================

vectorizer_path = os.path.join(
    MODELS_DIR,
    "tfidf_vectorizer_v2.pkl"
)

model_path = os.path.join(
    MODELS_DIR,
    "resume_classifier_v2.pkl"
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
# CREATE RESULT DATAFRAME
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
# ONLY WRONG PREDICTIONS
# ==================================================

errors = results[
    results["Actual_Category"]
    !=
    results["Predicted_Category"]
].copy()


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
    len(errors) /
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
print("MOST COMMON CATEGORY CONFUSIONS")
print("=" * 70)


print(
    confusions.head(15).to_string(
        index=False
    )
)


# ==================================================
# ERROR COUNT BY ACTUAL CATEGORY
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
print("ERRORS BY ACTUAL CATEGORY")
print("=" * 70)


print(
    category_errors.to_string(
        index=False
    )
)


# ==================================================
# SAVE ALL ERRORS
# ==================================================

errors_path = os.path.join(
    RESULTS_DIR,
    "classification_errors.csv"
)


errors.to_csv(
    errors_path,
    index=False
)


# ==================================================
# SAVE CONFUSIONS
# ==================================================

confusions_path = os.path.join(
    RESULTS_DIR,
    "category_confusions.csv"
)


confusions.to_csv(
    confusions_path,
    index=False
)


# ==================================================
# SAVE CATEGORY ERROR SUMMARY
# ==================================================

category_errors_path = os.path.join(
    RESULTS_DIR,
    "category_error_summary.csv"
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
    "Classification errors:"
)

print(
    errors_path
)


print(
    "\nCategory confusions:"
)

print(
    confusions_path
)


print(
    "\nCategory error summary:"
)

print(
    category_errors_path
)

print("\n")
print("=" * 70)
print("ERROR ANALYSIS COMPLETED")
print("=" * 70)