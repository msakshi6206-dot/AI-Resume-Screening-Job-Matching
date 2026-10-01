import pymupdf
import joblib

from preprocessing import preprocess_text
# ---------------------------------------
# 1. Resume PDF Path
# ---------------------------------------
pdf_path = r"uploads\my_resume.pdf"
# ---------------------------------------
# 2. Load Saved Models
# ---------------------------------------
vectorizer = joblib.load(
    r"models\tfidf_vectorizer.pkl"
)

model = joblib.load(
    r"models\resume_classifier.pkl"
)

# ---------------------------------------
# 3. Extract Text from PDF
# ---------------------------------------
print("Reading resume PDF...")

doc = pymupdf.open(pdf_path)

resume_text = ""

for page in doc:
    resume_text += page.get_text()

doc.close()


print("PDF text extraction completed.")

print("\nExtracted characters:", len(resume_text))

# ---------------------------------------
# 4. Check Extracted Text
# ---------------------------------------
if not resume_text.strip():

    print("No text found in the PDF.")

    exit()

print("\nFirst 500 characters:")
print(resume_text[:500])
# ---------------------------------------
# 5. Preprocess Resume
# ---------------------------------------
processed_text = preprocess_text(resume_text)

print("\nText preprocessing completed.")

# ---------------------------------------
# 6. Convert to TF-IDF
# ---------------------------------------
resume_vector = vectorizer.transform(
    [processed_text]
)

# ---------------------------------------
# 7. Predict Category
# ---------------------------------------
prediction = model.predict(
    resume_vector
)

# ---------------------------------------
# 8. Display Result
# ---------------------------------------
print("\n===================================")
print("       RESUME ANALYSIS RESULT")
print("===================================")

print("Predicted Category:", prediction[0])

print("===================================")