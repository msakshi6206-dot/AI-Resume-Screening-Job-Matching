import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Load processed dataset
input_path = r"dataset\Resume\Resume_processed.csv"

print("Loading processed dataset...")

df = pd.read_csv(input_path)

print("Total resumes:", len(df))


# Create TF-IDF Vectorizer
print("\nCreating TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.95
)

# Handle missing/empty processed resumes
df["Processed_Resume"] = df["Processed_Resume"].fillna("").astype(str)

df = df[df["Processed_Resume"].str.strip() != ""]

print("Usable resumes for TF-IDF:", len(df))

# Convert resume text into numerical vectors
X = vectorizer.fit_transform(df["Processed_Resume"])

print("TF-IDF conversion completed!")

print("\nTF-IDF Matrix Shape:", X.shape)

print("Number of features:", len(vectorizer.get_feature_names_out()))

print("\nFirst 20 features:")

print(vectorizer.get_feature_names_out()[:20])