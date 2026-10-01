import joblib
from preprocessing import preprocess_text

# Load saved models
vectorizer = joblib.load(r"models\tfidf_vectorizer.pkl")
model = joblib.load(r"models\resume_classifier.pkl")

# Sample resume
resume_text = """
Python Developer with experience in Machine Learning,
Pandas, NumPy, Scikit-learn, Flask and SQL.
Developed machine learning models and data analysis projects.
"""

# Preprocess resume
processed_text = preprocess_text(resume_text)

# Convert text into TF-IDF
resume_vector = vectorizer.transform([processed_text])

# Predict category
prediction = model.predict(resume_vector)

print("\nResume Category Prediction:")
print(prediction[0])