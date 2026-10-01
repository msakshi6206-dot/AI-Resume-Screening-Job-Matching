from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from resume_analyzer import analyze_resume


# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)

CORS(app)


# ==========================================
# UPLOAD CONFIGURATION
# ==========================================

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# HOME ROUTE
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message":
        "AI Resume Screening & Job Matching API is running!"
    })


# ==========================================
# RESUME ANALYSIS API
# ==========================================

@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        # --------------------------------------
        # CHECK RESUME FILE
        # --------------------------------------

        if "resume" not in request.files:

            return jsonify({
                "error": "Resume PDF is required."
            }), 400


        # --------------------------------------
        # GET UPLOADED FILE
        # --------------------------------------

        resume = request.files["resume"]


        # --------------------------------------
        # CHECK FILE NAME
        # --------------------------------------

        if resume.filename == "":

            return jsonify({
                "error":
                "Please select a resume PDF."
            }), 400


        # --------------------------------------
        # CHECK PDF EXTENSION
        # --------------------------------------

        if not resume.filename.lower().endswith(".pdf"):

            return jsonify({
                "error":
                "Only PDF files are supported."
            }), 400


        # --------------------------------------
        # GET JOB DESCRIPTION
        # --------------------------------------

        job_description = request.form.get(
            "job_description",
            ""
        )


        # --------------------------------------
        # CHECK JOB DESCRIPTION
        # --------------------------------------

        if not job_description.strip():

            return jsonify({
                "error":
                "Job description is required."
            }), 400


        # --------------------------------------
        # SAVE RESUME
        # --------------------------------------

        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            resume.filename
        )


        resume.save(file_path)


        # --------------------------------------
        # ANALYZE RESUME
        # --------------------------------------

        result = analyze_resume(
            file_path,
            job_description
        )


        # --------------------------------------
        # CHECK RESULT
        # --------------------------------------

        if result is None:

            return jsonify({
                "error":
                "Could not extract text from resume."
            }), 400


        # --------------------------------------
        # MAKE SURE RESULT IS DICTIONARY
        # --------------------------------------

        if not isinstance(result, dict):

            return jsonify({
                "error":
                "Invalid analysis result returned by resume_analyzer."
            }), 500


        # ======================================
        # DETECTED ROLE
        # ======================================

        # If resume_analyzer already returns
        # detected_role, keep it.

        if "detected_role" not in result:

            if "role" in result:

                result["detected_role"] = \
                    result["role"]

            elif "detectedRole" in result:

                result["detected_role"] = \
                    result["detectedRole"]

            else:

                result["detected_role"] = "-"


        # ======================================
        # PREDICTED CATEGORY
        # ======================================

        if "predicted_category" not in result:

            if "category" in result:

                result["predicted_category"] = \
                    result["category"]

            else:

                result["predicted_category"] = "-"


        # ======================================
        # MATCH SCORE
        # ======================================

        if "match_score" not in result:

            result["match_score"] = 0


        # ======================================
        # TEXT SIMILARITY
        # ======================================

        if "text_similarity" not in result:

            result["text_similarity"] = 0


        # ======================================
        # SKILL MATCH
        # ======================================

        if "skill_match" not in result:

            result["skill_match"] = 0


        # ======================================
        # MATCHED SKILLS
        # ======================================

        if "matched_skills" not in result:

            result["matched_skills"] = []


        # ======================================
        # MISSING SKILLS
        # ======================================

        if "missing_skills" not in result:

            result["missing_skills"] = []


        # ======================================
        # PRINT RESULT IN TERMINAL
        # ======================================

        print("\n========================================")
        print("RESUME ANALYSIS RESULT")
        print("========================================")

        print(
            "Detected Role:",
            result.get(
                "detected_role",
                "-"
            )
        )

        print(
            "Predicted Category:",
            result.get(
                "predicted_category",
                "-"
            )
        )

        print(
            "Match Score:",
            result.get(
                "match_score",
                0
            )
        )

        print(
            "Text Similarity:",
            result.get(
                "text_similarity",
                0
            )
        )

        print(
            "Skill Match:",
            result.get(
                "skill_match",
                0
            )
        )

        print(
            "Matched Skills:",
            result.get(
                "matched_skills",
                []
            )
        )

        print(
            "Missing Skills:",
            result.get(
                "missing_skills",
                []
            )
        )

        print("========================================\n")


        # ======================================
        # RETURN JSON RESPONSE
        # ======================================

        return jsonify(result)


    # ==========================================
    # ERROR HANDLING
    # ==========================================

    except Exception as e:

        print("\n========================================")
        print("ERROR IN /analyze")
        print("========================================")
        print(str(e))
        print("========================================\n")


        return jsonify({
            "error":
            "Resume analysis failed: " +
            str(e)
        }), 500


# ==========================================
# RUN FLASK SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )

