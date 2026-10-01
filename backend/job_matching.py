import joblib

from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_text
# ---------------------------------------
# 1. Load Saved TF-IDF Vectorizer
# ---------------------------------------
vectorizer = joblib.load(
    r"models\tfidf_vectorizer.pkl"
)

# ---------------------------------------
# 2. Sample Resume
# ---------------------------------------
resume_text = """
Python Developer with experience in Machine Learning,
Pandas, NumPy, Scikit-learn and Flask.
Developed machine learning models and data analysis projects.
Experience with SQL and Python programming.
"""
# ---------------------------------------
# 3. Sample Job Description
# ---------------------------------------
job_description = """
We are looking for a Python Developer with experience
in Python, Machine Learning, SQL, Pandas, Flask and
Scikit-learn.

The candidate should have knowledge of data analysis,
machine learning model development and Python programming.
"""
# ---------------------------------------
# 4. Preprocess Both Texts
# --------------------------------------
processed_resume = preprocess_text(resume_text)

processed_job = preprocess_text(job_description)

# ---------------------------------------
# 5. Convert Text to TF-IDF
# ---------------------------------------
resume_vector = vectorizer.transform(
    [processed_resume]
)

job_vector = vectorizer.transform(
    [processed_job]
)

# ---------------------------------------
# 6. Calculate Cosine Similarity
# ---------------------------------------
similarity = cosine_similarity(
    resume_vector,
    job_vector
)[0][0]

# Convert to percentage
match_score = similarity * 100

# ---------------------------------------
# 7. Display Result
# ---------------------------------------
print("\n===================================")
print("       JOB MATCHING RESULT")
print("===================================")

print(
    "Match Score:",
    round(match_score, 2),
    "%"
)

print("===================================")