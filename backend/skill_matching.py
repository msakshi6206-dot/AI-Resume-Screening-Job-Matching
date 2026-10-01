import re


# --------------------------------------------------
# Skill aliases
# --------------------------------------------------

SKILL_ALIASES = {

    # Programming
    "python": "python",
    "java": "java",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "c++": "c++",
    "c#": "c#",

    # Web Development
    "html": "html",
    "html5": "html",
    "css": "css",
    "css3": "css",
    "angular": "angular",
    "react": "react",
    "react.js": "react",
    "node.js": "node.js",
    "nodejs": "node.js",
    "express": "express",
    "express.js": "express",
    "flask": "flask",
    "django": "django",

    # Database
    "sql": "sql",
    "mysql": "mysql",
    "mongodb": "mongodb",
    "mongo db": "mongodb",

    # AI / ML
    "machine learning": "machine learning",
    "ml": "machine learning",
    "deep learning": "deep learning",
    "artificial intelligence": "artificial intelligence",
    "ai": "artificial intelligence",
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "nlp": "nlp",
    "natural language processing": "nlp",

    # Tools
    "git": "git",
    "github": "github",
    "docker": "docker",
    "aws": "aws",
    "azure": "azure",

    # Office / Productivity
    "excel": "excel",
    "microsoft excel": "excel",
    "ms excel": "excel",

    "microsoft office": "microsoft office",
    "ms office": "microsoft office",

    "word": "microsoft word",
    "microsoft word": "microsoft word",
    "ms word": "microsoft word",

    "powerpoint": "microsoft powerpoint",
    "microsoft powerpoint": "microsoft powerpoint",
    "ms powerpoint": "microsoft powerpoint",

    # Data Visualization
    "power bi": "power bi",
    "powerbi": "power bi",
    "tableau": "tableau",

    # Administrative Skills
    "communication": "communication",
    "verbal communication": "communication",
    "written communication": "communication",

    "organization": "organization",
    "organizational skills": "organization",

    "scheduling": "scheduling",
    "schedule management": "scheduling",

    "time management": "time management",

    "documentation": "documentation",
    "document management": "documentation",

    "administrative support": "administrative support",
    "administration": "administrative support",
    "administrative assistance": "administrative support",

    "record management": "record management",
    "records management": "record management",

    "appointment management": "appointment management",
    "appointment scheduling": "appointment management",

    "travel coordination": "travel coordination",
    "travel management": "travel coordination",

    "filing": "filing",
    "file management": "filing",

    "data entry": "data entry",

    "office management": "office management",

    "calendar management": "calendar management",

    "meeting coordination": "meeting coordination",
    "meeting scheduling": "meeting coordination",

    "email management": "email management",

    # General Professional Skills
    "customer service": "customer service",
    "problem solving": "problem solving",
    "problem-solving": "problem solving",
    "teamwork": "teamwork",
    "team work": "teamwork",
    "leadership": "leadership",
    "multitasking": "multitasking",
    "multi-tasking": "multitasking"
}


# --------------------------------------------------
# Normalize text
# --------------------------------------------------

def normalize_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Replace multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# Extract skills
# --------------------------------------------------

def extract_skills(text):

    text = normalize_text(text)

    found_skills = set()

    for skill, canonical_skill in SKILL_ALIASES.items():

        pattern = (
            r"(?<!\w)"
            + re.escape(skill)
            + r"(?!\w)"
        )

        if re.search(pattern, text):

            found_skills.add(canonical_skill)

    return sorted(found_skills)


# --------------------------------------------------
# Compare Resume and Job Description
# --------------------------------------------------

def compare_skills(resume_text, job_description):

    resume_skills = set(
        extract_skills(resume_text)
    )

    job_skills = set(
        extract_skills(job_description)
    )

    # Skills present in both
    matched_skills = sorted(
        resume_skills.intersection(job_skills)
    )

    # Required by job but not found in resume
    missing_skills = sorted(
        job_skills - resume_skills
    )

    return matched_skills, missing_skills