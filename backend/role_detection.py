import re


COMMON_JOB_TITLES = [
    "administrative assistant",
    "executive assistant",
    "office assistant",
    "office administrator",
    "administrative officer",
    "data entry operator",
    "data entry clerk",
    "customer service representative",
    "customer service executive",
    "sales executive",
    "sales manager",
    "marketing executive",
    "marketing manager",
    "business analyst",
    "data analyst",
    "data scientist",
    "machine learning engineer",
    "machine learning developer",
    "ai engineer",
    "artificial intelligence engineer",
    "software engineer",
    "software developer",
    "web developer",
    "frontend developer",
    "backend developer",
    "full stack developer",
    "python developer",
    "java developer",
    "angular developer",
    "react developer",
    "php developer",
    "database administrator",
    "database developer",
    "system administrator",
    "network administrator",
    "network engineer",
    "devops engineer",
    "cloud engineer",
    "project manager",
    "project coordinator",
    "hr manager",
    "hr executive",
    "human resources executive",
    "accountant",
    "financial analyst",
    "finance manager",
    "teacher",
    "lecturer",
    "professor",
    "graphic designer",
    "ui designer",
    "ux designer",
    "ui ux designer",
    "content writer",
    "technical writer",
    "consultant",
    "business development executive",
    "business development manager",
    "digital marketing executive",
    "chef",
    "nurse",
    "doctor",
    "pharmacist",
    "civil engineer",
    "mechanical engineer",
    "electrical engineer",
    "automobile engineer"
]


def detect_role(resume_text, job_description=""):

    if not isinstance(resume_text, str):
        resume_text = ""

    if not isinstance(job_description, str):
        job_description = ""

    resume_lower = resume_text.lower()
    job_lower = job_description.lower()

    # Normalize spaces
    resume_lower = re.sub(r"\s+", " ", resume_lower)
    job_lower = re.sub(r"\s+", " ", job_lower)

    # ------------------------------------------------
    # 1. Check resume first
    # ------------------------------------------------
    for title in COMMON_JOB_TITLES:

        if title in resume_lower:
            return title.title()

    # ------------------------------------------------
    # 2. Check job description
    # ------------------------------------------------
    for title in COMMON_JOB_TITLES:

        if title in job_lower:
            return title.title()

    # ------------------------------------------------
    # 3. Fallback: check first 15 resume lines
    # ------------------------------------------------
    lines = [
        line.strip()
        for line in resume_text.splitlines()
        if line.strip()
    ]

    keywords = [
        "assistant",
        "developer",
        "engineer",
        "manager",
        "analyst",
        "designer",
        "executive",
        "administrator",
        "consultant",
        "specialist",
        "coordinator",
        "officer",
        "teacher",
        "accountant",
        "scientist"
    ]

    for line in lines[:15]:

        clean_line = re.sub(
            r"[^a-zA-Z\s&/-]",
            "",
            line
        ).strip()

        words = clean_line.split()

        if 2 <= len(words) <= 7:

            lower_line = clean_line.lower()

            for keyword in keywords:

                if keyword in lower_line:
                    return clean_line.title()

    # ------------------------------------------------
    # 4. Nothing detected
    # ------------------------------------------------
    return "Not Detected"