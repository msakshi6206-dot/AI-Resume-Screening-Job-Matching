\# AI Resume Screening \& Job Matching



\## 📌 Overview



AI Resume Screening \& Job Matching is a machine learning based project that analyzes a candidate's resume and compares it with a given job description.



The system uses Natural Language Processing (NLP) and Machine Learning techniques to identify the resume category, detect the job role, compare skills, and calculate an overall job match score.



\## ✨ Features



\* Upload resume in PDF format

\* Extract text from resume

\* Predict resume category using Machine Learning

\* Detect job role from the job description

\* Match resume skills with required job skills

\* Identify matched and missing skills

\* Calculate job match score

\* Display text similarity and skill match score

\* Simple web-based user interface



\## 🛠️ Technologies Used



\* Python

\* Flask

\* HTML

\* CSS

\* JavaScript

\* Natural Language Processing (NLP)

\* TF-IDF

\* Linear SVM

\* Scikit-learn

\* PyMuPDF

\* Joblib



\## 🤖 Machine Learning Approach



The project uses:



\* \*\*TF-IDF\*\* for converting text into numerical features

\* \*\*Linear SVM\*\* for resume category classification

\* \*\*Cosine Similarity\*\* for comparing resume and job description text

\* \*\*Rule-based role detection\*\* for identifying common job roles

\* \*\*Skill matching\*\* for finding matched and missing skills



\## 📊 Job Match Score



The final job match score is calculated using:



\*\*40% Text Similarity + 60% Skill Match\*\*



The result also displays:



\* Text Similarity

\* Skill Match

\* Matched Skills

\* Missing Skills



\## 📁 Project Structure



```text

AI-Resume-Screening-Job-Matching/

│

├── backend/

│   ├── app.py

│   ├── resume\_analyzer.py

│   ├── preprocessing.py

│   ├── role\_detection.py

│   ├── skill\_matching.py

│   ├── job\_matching.py

│   └── other Python modules

│

├── frontend/

│   ├── index.html

│   ├── style.css

│   └── script.js

│

├── dataset/

│   └── Resume/

│

├── models/

│   └── trained ML models

│

├── results/

│

├── requirements.txt

├── .gitignore

└── README.md

```



\## ▶️ How It Works



1\. User uploads a resume PDF.

2\. The system extracts the resume text.

3\. The resume is preprocessed using NLP techniques.

4\. The trained machine learning model predicts the resume category.

5\. The system analyzes the job description.

6\. The job role is detected.

7\. Resume skills are compared with required skills.

8\. Text similarity and skill matching are calculated.

9\. The final job match score is displayed.



\## 🖥️ Output



The application displays:



\* Detected Role

\* Predicted Dataset Category

\* Job Match Score

\* Text Similarity

\* Skill Match

\* Matched Skills

\* Missing Skills



\## 🔮 Future Scope



\* Improve job role detection

\* Add more job roles and skills

\* Improve resume classification accuracy

\* Add advanced NLP techniques

\* Add resume recommendations

\* Deploy the application online

\* Add a more advanced recommendation system



\## 👩‍💻 Author



\*\*Sakshi Mistry\*\*



M.Sc. IT Student



This project was developed as a personal learning and portfolio project to apply concepts of Machine Learning, NLP, and web development.



