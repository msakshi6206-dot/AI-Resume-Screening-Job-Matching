import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Load English stopwords
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Clean and preprocess resume text.
    """

    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


def process_dataset():
    # Input dataset
    input_path = r"dataset\Resume\Resume_cleaned.csv"

    # Output dataset
    output_path = r"dataset\Resume\Resume_processed.csv"

    print("Loading dataset...")

    df = pd.read_csv(input_path)

    print("Total resumes:", len(df))

    print("\nApplying NLP preprocessing...")

    df["Processed_Resume"] = df["Resume_str"].apply(preprocess_text)

    df.to_csv(output_path, index=False)

    print("\nProcessing completed successfully!")
    print("Saved to:", output_path)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst processed resume:")
    print(df["Processed_Resume"].iloc[0])


if __name__ == "__main__":
    process_dataset()