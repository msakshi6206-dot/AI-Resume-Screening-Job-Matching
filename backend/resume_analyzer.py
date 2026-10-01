import os
import re
import fitz
import joblib

from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess_text
from skill_matching import compare_skills


# --------------------------------------------------
# BASE DIRECTORY
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------

vectorizer_path = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer_v3.pkl"
)

model_path = os.path.join(
    BASE_DIR,
    "models",
    "resume_classifier_v3.pkl"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)


# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_text_from_pdf(pdf_path):

    text = ""

    try:
        document = fitz.open(pdf_path)

        for page in document:
            text += page.get_text()

        document.close()

    except Exception as e:
        print("PDF extraction error:", e)
        return ""

    return text


# --------------------------------------------------
# ROLE DETECTION
# --------------------------------------------------

def detect_role(job_description):

    text = job_description.lower().strip()

    # Common job roles
    role_patterns = [
        (
            r"\badministrative assistant\b|\badmin assistant\b",
            "Administrative Assistant"
        ),
        (
            r"\bsoftware developer\b|\bsoftware engineer\b",
            "Software Developer"
        ),
        (
            r"\bweb developer\b",
            "Web Developer"
        ),
        (
            r"\bfrontend developer\b|\bfront end developer\b",
            "Frontend Developer"
        ),
        (
            r"\bbackend developer\b|\bback end developer\b",
            "Backend Developer"
        ),
        (
            r"\bfull stack developer\b|\bfullstack developer\b",
            "Full Stack Developer"
        ),
        (
            r"\bdata analyst\b",
            "Data Analyst"
        ),
        (
            r"\bdata scientist\b",
            "Data Scientist"
        ),
        (
            r"\bmachine learning engineer\b|\bml engineer\b",
            "Machine Learning Engineer"
        ),
        (
            r"\bai engineer\b|\bartificial intelligence engineer\b",
            "AI Engineer"
        ),
        (
            r"\bproject manager\b",
            "Project Manager"
        ),
        (
            r"\bhr manager\b|\bhuman resources manager\b",
            "HR Manager"
        ),
        (
            r"\bhr executive\b|\bhuman resources executive\b",
            "HR Executive"
        ),
        (
            r"\baccountant\b",
            "Accountant"
        ),
        (
            r"\bmarketing executive\b",
            "Marketing Executive"
        ),
        (
            r"\bmarketing manager\b",
            "Marketing Manager"
        ),
        (
            r"\bsales executive\b",
            "Sales Executive"
        ),
        (
            r"\bcustomer service representative\b|\bcustomer support\b",
            "Customer Support"
        ),
        (
            r"\bgraphic designer\b",
            "Graphic Designer"
        ),
        (
            r"\bui/ux designer\b|\bui ux designer\b",
            "UI/UX Designer"
        ),
        (
            r"\bteacher\b|\blecturer\b",
            "Teacher"
        ),
        (
            r"\bchef\b",
            "Chef"
        )
    ]

    for pattern, role in role_patterns:

        if re.search(pattern, text):
            return role

    # --------------------------------------------------
    # Try to detect role from common JD phrases
    # --------------------------------------------------

    patterns = [
        r"job title\s*[:\-]\s*([A-Za-z][A-Za-z /&-]{2,60})",
        r"position\s*[:\-]\s*([A-Za-z][A-Za-z /&-]{2,60})",
        r"role\s*[:\-]\s*([A-Za-z][A-Za-z /&-]{2,60})",
        r"designation\s*[:\-]\s*([A-Za-z][A-Za-z /&-]{2,60})"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            job_description,
            re.IGNORECASE
        )

        if match:

            role = match.group(1).strip()

            role = re.split(
                r"[\n\r.,;]",
                role
            )[0].strip()

            if 2 <= len(role.split()) <= 7:
                return role.title()

    return "Not detected"


# --------------------------------------------------
# SKILL MATCHING HELPER
# --------------------------------------------------

def get_skill_results(resume_text, job_description):

    result = compare_skills(
        resume_text,
        job_description
    )

    # Support tuple return
    if isinstance(result, tuple):

        matched_skills = result[0]
        missing_skills = result[1]

    # Support dictionary return
    elif isinstance(result, dict):

        matched_skills = result.get(
            "matched_skills",
            []
        )

        missing_skills = result.get(
            "missing_skills",
            []
        )

    else:

        matched_skills = []
        missing_skills = []

    return matched_skills, missing_skills


# --------------------------------------------------
# MAIN ANALYSIS FUNCTION
# --------------------------------------------------

def analyze_resume(pdf_path, job_description):

    # --------------------------------------------------
    # Extract resume text
    # --------------------------------------------------

    resume_text = extract_text_from_pdf(pdf_path)

    if not resume_text.strip():
        return None

    # --------------------------------------------------
    # Preprocess text
    # --------------------------------------------------

    processed_resume = preprocess_text(
        resume_text
    )

    processed_job = preprocess_text(
        job_description
    )

    if not processed_resume.strip():
        return None

    if not processed_job.strip():
        return None

    # --------------------------------------------------
    # Dataset category prediction
    # --------------------------------------------------

    resume_vector = vectorizer.transform(
        [processed_resume]
    )

    predicted_category = model.predict(
        resume_vector
    )[0]

    # --------------------------------------------------
    # Text similarity
    # --------------------------------------------------

    job_vector = vectorizer.transform(
        [processed_job]
    )

    text_similarity = cosine_similarity(
        resume_vector,
        job_vector
    )[0][0]

    text_similarity_score = round(
        text_similarity * 100,
        2
    )

    # --------------------------------------------------
    # Skill matching
    # --------------------------------------------------

    matched_skills, missing_skills = get_skill_results(
        resume_text,
        job_description
    )

    total_required_skills = (
        len(matched_skills)
        + len(missing_skills)
    )

    if total_required_skills > 0:

        skill_match_score = (
            len(matched_skills)
            / total_required_skills
        ) * 100

    else:

        skill_match_score = 0

    skill_match_score = round(
        skill_match_score,
        2
    )

    # --------------------------------------------------
    # FINAL SCORE
    #
    # 40% Text Similarity
    # 60% Skill Match
    # --------------------------------------------------

    final_score = (
        (text_similarity_score * 0.40)
        +
        (skill_match_score * 0.60)
    )

    final_score = round(
        final_score,
        2
    )

    # --------------------------------------------------
    # Detect actual job role
    # --------------------------------------------------

    detected_role = detect_role(
        job_description
    )

    # --------------------------------------------------
    # Return complete result
    # --------------------------------------------------

    return {

        "detected_role": detected_role,

        "predicted_category": predicted_category,

        "text_similarity": text_similarity_score,

        "skill_match": skill_match_score,

        "match_score": final_score,

        "matched_skills": matched_skills,

        "missing_skills": missing_skills
    }