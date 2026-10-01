const API_URL = "http://127.0.0.1:5000/analyze";


// ========================================
// GET HTML ELEMENTS
// ========================================

const resumeInput = document.getElementById("resume");

const jobDescriptionInput =
    document.getElementById("jobDescription");

const analyzeBtn =
    document.getElementById("analyzeBtn");

const loading =
    document.getElementById("loading");

const errorBox =
    document.getElementById("error");

const resultSection =
    document.getElementById("result");

const detectedRole =
    document.getElementById("detectedRole");

const category =
    document.getElementById("category");

const matchScore =
    document.getElementById("matchScore");

const scoreProgress =
    document.getElementById("scoreProgress");

const scoreMessage =
    document.getElementById("scoreMessage");

const textSimilarity =
    document.getElementById("textSimilarity");

const skillMatch =
    document.getElementById("skillMatch");

const matchedSkills =
    document.getElementById("matchedSkills");

const missingSkills =
    document.getElementById("missingSkills");


// ========================================
// ANALYZE RESUME
// ========================================

analyzeBtn.addEventListener(
    "click",
    async function () {

        // Hide previous result/error
        errorBox.style.display = "none";
        resultSection.style.display = "none";


        // --------------------------------
        // Check Resume
        // --------------------------------

        if (
            !resumeInput.files ||
            resumeInput.files.length === 0
        ) {

            showError(
                "Please select a resume PDF."
            );

            return;
        }


        const resumeFile =
            resumeInput.files[0];


        // --------------------------------
        // Check PDF
        // --------------------------------

        if (
            !resumeFile.name
                .toLowerCase()
                .endsWith(".pdf")
        ) {

            showError(
                "Only PDF files are supported."
            );

            return;
        }


        // --------------------------------
        // Get Job Description
        // --------------------------------

        const jobDescription =
            jobDescriptionInput.value.trim();


        if (!jobDescription) {

            showError(
                "Please enter the job description."
            );

            return;
        }


        // --------------------------------
        // Create Form Data
        // --------------------------------

        const formData = new FormData();

        formData.append(
            "resume",
            resumeFile
        );

        formData.append(
            "job_description",
            jobDescription
        );


        // --------------------------------
        // Show Loading
        // --------------------------------

        loading.style.display = "flex";

        analyzeBtn.disabled = true;

        analyzeBtn.innerHTML =
            "⏳ Analyzing Resume...";


        try {

            // --------------------------------
            // Send Request To Flask
            // --------------------------------

            const response =
                await fetch(
                    API_URL,
                    {
                        method: "POST",
                        body: formData
                    }
                );


            // --------------------------------
            // Get Backend Response
            // --------------------------------

            const data =
                await response.json();


            // --------------------------------
            // Check Error
            // --------------------------------

            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Resume analysis failed."
                );
            }


            // --------------------------------
            // Show Results
            // --------------------------------

            displayResults(data);


        } catch (error) {

            console.error(
                "Analysis Error:",
                error
            );

            showError(
                error.message ||
                "Unable to connect to backend."
            );


        } finally {

            // --------------------------------
            // Stop Loading
            // --------------------------------

            loading.style.display = "none";

            analyzeBtn.disabled = false;

            analyzeBtn.innerHTML =
                "🔍 Analyze Resume";
        }

    }
);


// ========================================
// DISPLAY RESULTS
// ========================================

function displayResults(data) {


    // --------------------------------
    // Detected Role
    // --------------------------------

    detectedRole.textContent =
        data.detected_role ||
        "Not detected";


    // --------------------------------
    // Predicted Category
    // --------------------------------

    category.textContent =
        data.predicted_category ||
        "Not detected";


    // --------------------------------
    // Final Match Score
    // --------------------------------

    const finalScore =
        Number(data.match_score || 0);


    matchScore.textContent =
        finalScore.toFixed(2) + "%";


    // --------------------------------
    // Progress Bar
    // --------------------------------

    const progress =
        Math.max(
            0,
            Math.min(
                100,
                finalScore
            )
        );


    scoreProgress.style.width =
        progress + "%";


    // --------------------------------
    // Score Message
    // --------------------------------

    if (finalScore >= 80) {

        scoreMessage.textContent =
            "Strong match between the resume and job description.";

    }

    else if (finalScore >= 60) {

        scoreMessage.textContent =
            "Good match with some skills or requirements to improve.";

    }

    else if (finalScore >= 40) {

        scoreMessage.textContent =
            "Moderate match. Some important requirements may be missing.";

    }

    else {

        scoreMessage.textContent =
            "Low match. The resume may need improvement for this job.";

    }


    // --------------------------------
    // Text Similarity
    // --------------------------------

    const similarity =
        Number(
            data.text_similarity || 0
        );


    textSimilarity.textContent =
        similarity.toFixed(2) + "%";


    // --------------------------------
    // Skill Match
    // --------------------------------

    const skillScore =
        Number(
            data.skill_match || 0
        );


    skillMatch.textContent =
        skillScore.toFixed(2) + "%";


    // --------------------------------
    // Matched Skills
    // --------------------------------

    displaySkills(
        matchedSkills,
        data.matched_skills,
        "No matched skills found."
    );


    // --------------------------------
    // Missing Skills
    // --------------------------------

    displaySkills(
        missingSkills,
        data.missing_skills,
        "No missing skills found."
    );


    // --------------------------------
    // Show Result Section
    // --------------------------------

    resultSection.style.display =
        "block";


    // --------------------------------
    // Scroll To Result
    // --------------------------------

    setTimeout(
        function () {

            resultSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        },
        100
    );
}


// ========================================
// DISPLAY SKILLS
// ========================================

function displaySkills(
    element,
    skills,
    emptyMessage
) {

    // Clear old skills
    element.innerHTML = "";


    // --------------------------------
    // Check Skills
    // --------------------------------

    if (
        !Array.isArray(skills) ||
        skills.length === 0
    ) {

        const li =
            document.createElement("li");

        li.textContent =
            emptyMessage;

        element.appendChild(li);

        return;
    }


    // --------------------------------
    // Add Skills
    // --------------------------------

    skills.forEach(
        function (skill) {

            const li =
                document.createElement("li");

            li.textContent =
                skill;

            element.appendChild(li);

        }
    );
}


// ========================================
// SHOW ERROR
// ========================================

function showError(message) {

    errorBox.textContent =
        message;

    errorBox.style.display =
        "block";


    errorBox.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });
}


// ========================================
// FILE SELECT
// ========================================

resumeInput.addEventListener(
    "change",
    function () {

        if (
            resumeInput.files &&
            resumeInput.files.length > 0
        ) {

            console.log(
                "Selected Resume:",
                resumeInput.files[0].name
            );
        }

    }
);