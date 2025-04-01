import streamlit as st
import fitz  # PyMuPDF
import requests
import json
import re  # For cleaning API response
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 Replace with your Gemini API key
GEMINI_API_KEY = "AIzaSyC0tsffQxhK2nwrj2Auaxelp6os0xjHE4U"




class ATSMatcher:
    def __init__(self, job_data_json):
        """
        Initialize the ATSMatcher with job data from a JSON object.
        :param job_data_json: JSON object containing job data (similar to a MongoDB document).
        """
        self.job_df = pd.DataFrame(job_data_json)
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def calculate_ats_score(self, skills, experience, cgpa, job_role=None):
        """
        Calculates the ATS score based on skills, experience, and CGPA.

        :param skills: List of candidate's skills.
        :param experience: Candidate's years of experience.
        :param cgpa: Candidate's CGPA.
        :param job_role: (Optional) Specific job role to filter jobs.
        :return: DataFrame with top 5 matching jobs sorted by ATS score.
        """
        if self.job_df.empty:
            return "No job listings available."

        # Filter jobs by role if specified
        job_df = self.job_df.copy()

        if job_df.empty:
            return f"No jobs found for the role '{job_role}'."

        # Preprocess candidate skills
        skills_text = " ".join(skills).lower()

        # TF-IDF Vectorization for job descriptions
        job_descriptions = job_df["Job Description"].fillna("").tolist()
        job_descriptions.append(skills_text)  # Append candidate's skills as a pseudo-description
        tfidf_matrix = self.vectorizer.fit_transform(job_descriptions)

        # Compute similarity scores
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

        # Normalize similarity scores (scale between 0-100)
        similarity_scores = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min() + 1e-5) * 100

        # Experience and CGPA matching (Scaled for ATS Score Calculation)
        job_df["Experience Score"] = 100 - abs(job_df["Experience"] - experience) * 10  # Penalizing large differences
        job_df["CGPA Score"] = 100 - abs(job_df["CGPA"] - cgpa) * 20  # Penalizing large differences

        # Normalize Experience and CGPA scores (Scale between 0-100)
        job_df["Experience Score"] = job_df["Experience Score"].clip(0, 100)
        job_df["CGPA Score"] = job_df["CGPA Score"].clip(0, 100)

        # Final ATS Score Calculation (Weighted)
        job_df["ATS Score"] = (0.6 * similarity_scores) + (0.1 * job_df["Experience Score"]) + (0.09 * job_df["CGPA Score"])

        # Sort and return top 5 matches
        top_matches = job_df[["ID", "Job Role", "ATS Score"]].sort_values(
            by="ATS Score", ascending=False
        )

        return top_matches



# input data
job_data_json = [
    {
        "ID": 1,
        "Job Role": "Full-Stack Developer",
        "Job Description": "Frontend Development using HTML, CSS, JavaScript, React.js. Backend development with Node.js, Python, and MongoDB.",
        "Skills": ["HTML", "CSS", "JavaScript", "React.js", "Node.js", "MongoDB"],
        "Experience": 2,
        "CGPA": 7.5
    },
    {
        "ID": 2,
        "Job Role": "Data Scientist",
        "Job Description": "Analyze large datasets using Python, Machine Learning, and AI techniques.",
        "Skills": ["Python", "Scikit-Learn", "pandas", "numpy", "TensorFlow", "pytorch"],
        "Experience": 3,
        "CGPA": 8.0
    },
    {
        "ID": 3,
        "Job Role": "UI/UX Designer",
        "Job Description": "Design user-friendly interfaces using Figma, CSS, and UX principles.",
        "Skills": ["Figma", "UX Design", "UI", "CSS"],
        "Experience": 1,
        "CGPA": 6.5
    },
    {
        "ID": 4,
        "Job Role": "Cloud Engineer",
        "Job Description": "Manage cloud infrastructure using AWS, Kubernetes, and Docker.",
        "Skills": ["AWS", "Kubernetes", "Docker", "Cloud Security"],
        "Experience": 4,
        "CGPA": 7.8
    },
    {
        "ID": 5,
        "Job Role": "Cybersecurity Analyst",
        "Job Description": "Perform cybersecurity analysis and penetration testing.",
        "Skills": ["Cybersecurity", "Penetration Testing", "Network Security"],
        "Experience": 5,
        "CGPA": 8.2
    },
    {
      "ID": 6,
      "Job Role": "Frontend Web Developer",
      "Job Description": "Develop responsive and interactive web applications using React.js, HTML, CSS, and JavaScript. Optimize performance, ensure cross-browser compatibility, and collaborate with backend developers for seamless integration.",
      "Required Skills": ["React.js", "JavaScript", "HTML", "CSS", "Redux", "TypeScript", "Next.js", "Tailwind CSS"],
      "Experience": 2,
      "CGPA": 7.0
    },
{
      "ID": 7,
      "Job Role": "Frontend Web Developer",
      "Job Description": "We are looking for a Frontend Web Developer with expertise in JavaScript, React.js, HTML, and CSS to build responsive and high-performance web applications. Experience with API integration, state management, and modern UI/UX practices is a plus.",
      "Required Skills": ["JavaScript", "HTML", "CSS", "Tailwind CSS"],
      "Experience": 2,
      "CGPA": 7.0
    }
]


ats_matcher=ATSMatcher(job_data_json)
jobrole=ats_matcher.job_df["Job Role"].unique()
jobrole[0]=None



def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    bytes_data = pdf_file.read()
    doc = fitz.open(stream=bytes_data, filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text("text")

    return text


def clean_json_response(api_response):
    """Sanitize API response and extract valid JSON."""
    try:
        # Remove Markdown formatting (e.g., ```json ... ```)
        cleaned_text = re.sub(r"```json|```", "", api_response).strip()

        # Convert to valid JSON format
        return json.loads(cleaned_text)

    except json.JSONDecodeError:
        return {"error": "Invalid JSON format received from API"}


def get_resume_data_from_gemini(resume_text):
    """Use Gemini API to extract structured resume details."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    headers = {"Content-Type": "application/json"}

    prompt = f"""
    Extract structured resume details from the given text.
    The output should be in valid JSON format with fields:

    {{
        "Name": "",
        "Summary": "",
        "Skills": []
        "Education": [
        ],
        "Work Experience": [
        ],
        "Projects": [
        ],
        "Achievements": [],
        "Certifications": [],
        "CGPA": ""
    }}

    Resume Text:
    {resume_text}
    """

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        if "candidates" in result and len(result["candidates"]) > 0:
            api_response_text = result["candidates"][0]["content"]["parts"][0]["text"]
            return clean_json_response(api_response_text)
        else:
            return {"error": "No valid response from Gemini API"}
    else:
        return {"error": f"API Error {response.status_code}: {response.text}"}



# for feedbacks
import requests
import json
import re  # For cleaning API response

def clean_json_response(api_response):
    """Sanitize API response and extract valid JSON."""
    try:
        cleaned_text = re.sub(r"```json|```", "", api_response).strip()
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format received from API"}


def get_resume_feedback(resume_text):
    """Use Gemini API to analyze resume and provide feedback."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    # Escaping curly braces in f-string to prevent format errors
    prompt = f"""
    Analyze the given resume text and provide structured feedback.
    The output should be in valid JSON format with the following fields:

    {{
        "Strengths": [],
        "Weaknesses": [],
        "Suggestions": [],
        "Overall Assessment": "Weak / Medium / Strong / Excellent"
    }}

    Resume Text:
    {resume_text}
    """

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise error for non-200 status codes

        result = response.json()

        # Gemini API's response might have a different structure
        if "candidates" in result and result["candidates"]:
            api_response_text = result["candidates"][0]["content"]["parts"][0]["text"]
            return clean_json_response(api_response_text)
        else:
            return {"error": "Unexpected response format from Gemini API"}

    except requests.exceptions.RequestException as e:
        return {"error": f"API Request Failed: {str(e)}"}

    except KeyError:
        return {"error": "Invalid response format from API"}


st.header("ATS System")

# File Uploader
uploadFile = st.file_uploader("Upload Your Resume (PDF)", type=['pdf'])

structured_data=[]

if uploadFile:
    extracted_text = extract_text_from_pdf(uploadFile)
    st.text_area("Extracted Resume Text", extracted_text, height=300)

    st.subheader("Extracting Resume Details..")
    structured_data = get_resume_data_from_gemini(extracted_text)

    st.success("Extracting the Details Successfully")
    # st.json(type(structured_data["Skills"]))


    opt = st.selectbox("Select a Job Role", options=jobrole)
    btn = st.button("Calculate The ATS Score")
    if btn:
        cgpa= structured_data["Cgpa"] if structured_data["CGPA"] else 0
        exp=structured_data["Work Experience"] if structured_data["Work Experience"] else 0
        skills=structured_data["Skills"] if structured_data["Skills"] else []
        topmatches=ats_matcher.calculate_ats_score(skills,exp,cgpa,opt)
        if opt:
            st.write(topmatches[topmatches["Job Role"] == opt])
        st.write(topmatches)

    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = None  # Store feedback results persistently

    btn = st.button("Check For Feedbacks")

    if btn:
        data = get_resume_feedback(extracted_text)
        st.session_state.feedback_data = data  # Store the feedback in session state

    # Display feedback only if available
    if st.session_state.feedback_data:
        st.subheader("Resume Feedback")
        st.json(st.session_state.feedback_data)
