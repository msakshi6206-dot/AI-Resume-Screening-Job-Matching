import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------
input_path = r"dataset\Resume\Resume_processed.csv"

print("Loading processed dataset...")

df = pd.read_csv(input_path)

print("Original resumes:", len(df))

# ---------------------------------------
# 2. Clean Data
# ---------------------------------------
df["Processed_Resume"] = df["Processed_Resume"].fillna("").astype(str)

df = df[df["Processed_Resume"].str.strip() != ""]

print("Usable Resumes:", len(df))

print("Number of categories:", df["Category"].nunique())

# ---------------------------------------
# 3. Separate Input and Target
# ---------------------------------------
X_text = df["Processed_Resume"]

y = df["Category"]

# ---------------------------------------
# 4. Split Dataset FIRST
# ---------------------------------------
print("\nSplitting dataset into train and test sets...")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train set size:", len(X_train_text))
print("Test set size:", len(X_test_text))

# ---------------------------------------
# 5. Create TF-IDF
# ---------------------------------------
print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.95
)

# IMPORTANT:
# Fit TF-IDF ONLY on training data
X_train = vectorizer.fit_transform(X_train_text)

X_test = vectorizer.transform(X_test_text)

print("Training TF-IDF shape:", X_train.shape)

print("Testing TF-IDF shape:", X_test.shape)

# ---------------------------------------
# 6. Train Logistic Regression
# ---------------------------------------
print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

print("Model training completed.")

# ---------------------------------------
# 7. Save Models
# ---------------------------------------
joblib.dump(
    vectorizer,
    r"models\tfidf_vectorizer.pkl"
)

joblib.dump(
    model,
    r"models\resume_classifier.pkl"
)

print("TF-IDF vectorizer saved.")

print("Classification model saved.")

# ---------------------------------------
# 8. Make Predictions
# ---------------------------------------
print("\nMaking predictions...")

y_pred = model.predict(X_test)

# ---------------------------------------
# 9. Accuracy
# ---------------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# ---------------------------------------
# 10. Classification Report
# ---------------------------------------
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)