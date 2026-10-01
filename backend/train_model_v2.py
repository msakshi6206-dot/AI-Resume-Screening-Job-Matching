import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# ==================================================
# PATHS
# =================================================
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

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

# ==================================================
# LOAD DATASET
# ==================================================
print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(
    DATASET_PATH
)

print(
    "Original resumes:",
    len(df)
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

# Remove empty resumes
df = df[
    df["Processed_Resume"].str.strip() != ""
].copy()

print(
    "Usable resumes:",
    len(df)
)

print(
    "Number of categories:",
    df["Category"].nunique()
)

# ==================================================
# FEATURES AND TARGET
# ==================================================
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

print(
    "Training set:",
    len(X_train)
)

print(
    "Testing set:",
    len(X_test)
)

# ==================================================
# TF-IDF VECTORIZER
# ==================================================
print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(

    # Single words + two-word phrases
    ngram_range=(1, 2),

    # Maximum vocabulary
    max_features=10000,

    # Ignore very rare words
    min_df=2,

    # Ignore extremely common words
    max_df=0.95,

    # Give more importance to informative terms
    sublinear_tf=True
)

# IMPORTANT:
# Fit ONLY on training data
X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print(
    "Training TF-IDF shape:",
    X_train_tfidf.shape
)

print(
    "Testing TF-IDF shape:",
    X_test_tfidf.shape
)

# ==================================================
# LINEAR SVM MODEL
# ==================================================
print("\nTraining Linear SVM...")

model = LinearSVC(

    C=1.0,

    class_weight="balanced",

    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
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
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(
    f"Model Accuracy: {accuracy * 100:.2f}%"
)

# ==================================================
# CLASSIFICATION REPORT
# ==================================================
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)
# ==================================================
# SAVE V2 MODEL
#==================================================
vectorizer_path = os.path.join(
    MODELS_DIR,
    "tfidf_vectorizer_v2.pkl"
)

model_path = os.path.join(
    MODELS_DIR,
    "resume_classifier_v2.pkl"
)

joblib.dump(
    vectorizer,
    vectorizer_path
)

joblib.dump(
    model,
    model_path
)

print("\n")
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    "Vectorizer:",
    vectorizer_path
)

print(
    "Model:",
    model_path
)
print("\nTraining completed successfully.")