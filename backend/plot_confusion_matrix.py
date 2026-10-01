import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==================================================
# LOAD DATASET
# ==================================================

print("=" * 60)
print("CONFUSION MATRIX GENERATION")
print("=" * 60)

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
# CATEGORY LABELS
# ==================================================

labels = sorted(
    y.unique()
)


# ==================================================
# CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)


# ==================================================
# DISPLAY
# ==================================================

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
    "Confusion Matrix - Resume Classification V2",
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
# SAVE IMAGE
# ==================================================
output_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix_v2.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nConfusion matrix saved successfully:")

print(
    output_path
)

plt.show()