import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


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

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# ==================================================
# LOAD DATA
# ==================================================

print("=" * 70)
print("V3 MODEL EXPERIMENT")
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


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ==================================================
# TF-IDF
# ==================================================

print("\nCreating TF-IDF features...")


vectorizer = TfidfVectorizer(

    ngram_range=(1, 2),

    max_features=10000,

    min_df=2,

    max_df=0.95,

    sublinear_tf=True
)


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
# TEST DIFFERENT C VALUES
# ==================================================

C_VALUES = [
    0.5,
    1.0,
    1.5,
    2.0
]


results = []


print("\n")
print("=" * 70)
print("TESTING DIFFERENT SVM C VALUES")
print("=" * 70)


for C in C_VALUES:

    print(
        f"\nTraining Linear SVM with C = {C}"
    )


    model = LinearSVC(

        C=C,

        class_weight="balanced",

        random_state=42
    )


    model.fit(
        X_train_tfidf,
        y_train
    )


    predictions = model.predict(
        X_test_tfidf
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    results.append({

        "C": C,

        "Accuracy": accuracy

    })


    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )


# ==================================================
# FIND BEST MODEL
# ==================================================

results_df = pd.DataFrame(
    results
)


best_index = (
    results_df["Accuracy"]
    .idxmax()
)


best_C = (
    results_df
    .loc[
        best_index,
        "C"
    ]
)


best_accuracy = (
    results_df
    .loc[
        best_index,
        "Accuracy"
    ]
)


# ==================================================
# PRINT COMPARISON
# ==================================================

print("\n")
print("=" * 70)
print("V3 EXPERIMENT RESULTS")
print("=" * 70)


for _, row in results_df.iterrows():

    print(
        f"C = {row['C']:<4} "
        f"Accuracy = "
        f"{row['Accuracy'] * 100:.2f}%"
    )


print("\n")
print(
    f"Best C value: {best_C}"
)

print(
    f"Best Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)


# ==================================================
# TRAIN FINAL V3 MODEL
# ==================================================

print("\n")
print("=" * 70)
print("TRAINING FINAL V3 MODEL")
print("=" * 70)


final_model = LinearSVC(

    C=best_C,

    class_weight="balanced",

    random_state=42
)


final_model.fit(
    X_train_tfidf,
    y_train
)


# ==================================================
# SAVE MODEL
# ==================================================

vectorizer_path = os.path.join(
    MODELS_DIR,
    "tfidf_vectorizer_v3.pkl"
)

model_path = os.path.join(
    MODELS_DIR,
    "resume_classifier_v3.pkl"
)


joblib.dump(
    vectorizer,
    vectorizer_path
)

joblib.dump(
    final_model,
    model_path
)


# ==================================================
# SAVE EXPERIMENT RESULTS
# ==================================================

results_path = os.path.join(
    BASE_DIR,
    "results",
    "v3_experiment_results.csv"
)


os.makedirs(
    os.path.dirname(results_path),
    exist_ok=True
)


results_df.to_csv(
    results_path,
    index=False
)


# ==================================================
# FINAL
# ==================================================

print("\n")
print("=" * 70)
print("V3 MODEL SAVED")
print("=" * 70)


print(
    "Vectorizer:"
)

print(
    vectorizer_path
)


print(
    "\nModel:"
)

print(
    model_path
)


print(
    "\nExperiment results:"
)

print(
    results_path
)


print("\n")
print("=" * 70)
print("V3 TRAINING COMPLETED")
print("=" * 70)