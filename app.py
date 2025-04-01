from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import requests
import json
import re  # For cleaning API response
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


# class function for ats score
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


    def calculate_ats_scoreMERN(self, skills, cgpa, job_role=None):
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
        job_descriptions = job_df["jobDescription"].fillna("").tolist()
        job_descriptions.append(skills_text)  # Append candidate's skills as a pseudo-description
        tfidf_matrix = self.vectorizer.fit_transform(job_descriptions)

        # Compute similarity scores
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

        # Normalize similarity scores (scale between 0-100)
        similarity_scores = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min() + 1e-5) * 100

        # Experience and CGPA matching (Scaled for ATS Score Calculation)  # Penalizing large differences
        job_df["CGPA Score"] = 100 - abs(job_df["cgpa"] - cgpa) * 20
        job_df["CGPA Score"] = job_df["CGPA Score"].clip(0, 100)

        # Final ATS Score Calculation (Weighted)
        job_df["ATS Score"] = (0.6 * similarity_scores) + (0.2 * job_df["CGPA Score"])

        # Sort and return top 5 matches
        top_matches = job_df[["_id", "jobRole", "ATS Score"]].sort_values(
            by="ATS Score", ascending=False
        )

        return top_matches

# all function to be written here
# function for extracting the text from the file
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    bytes_data = pdf_file.read()
    doc = fitz.open(stream=bytes_data, filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# to make the text json clear
def clean_json_response(api_response):
    """Sanitize API response and extract valid JSON."""
    try:
        # Remove Markdown formatting (e.g., ```json ... ```)
        cleaned_text = re.sub(r"```json|```", "", api_response).strip()

        # Convert to valid JSON format
        return json.loads(cleaned_text)

    except json.JSONDecodeError:
        return {"error": "Invalid JSON format received from API"}



# extracting the structured data from the text

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






# routes are defined
@app.route('/')
def home():
    return render_template("index.html")

structured_data=[]
@app.route("/upload", methods=['POST'])
def uploadResume():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded","success" : False}), 400

    file = request.files['resume']

    if file.filename == '':
        return jsonify({"error": "No file selected","success": False}), 400

    text=extract_text_from_pdf(file)
    text=get_resume_data_from_gemini(text)
    with open("resume_details.json", "w") as f:
        json.dump(text, f, indent=4)
    # print(structured_data)
    return jsonify({"text": text,"success" : True})


    # Save file (optional, adjust path as needed)
    # return jsonify({"message": "File uploaded successfully", "filename": file.filename,"success": True})

print(structured_data)



@app.route("/job")
def jobprotal():
    return render_template("jobUpload.html")



@app.route('/upload_job', methods=['POST'])
def upload_job():
    try:
        # Get form data
        job_id = request.form.get("job_id")
        job_role = request.form.get("job_role")
        job_description = request.form.get("job_description")
        skills = request.form.get("skills")  # Comma-separated values
        experience = request.form.get("experience")
        cgpa = request.form.get("cgpa")

        # Convert skills to a list
        skills_list = [skill.strip() for skill in skills.split(",")]

        # Create job dictionary
        job_data = {
            "ID": int(job_id),
            "Job Role": job_role,
            "Job Description": job_description,
            "Skills": skills_list,
            "Experience": float(experience),
            "CGPA": float(cgpa)
        }

        # Save job data to a JSON file (or you can store it in a database)
        with open("jobs.json", "a") as file:
            file.write(json.dumps(job_data) + "\n")



        return jsonify({"success": True, "message": "Job uploaded successfully!", "job": job_data})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400





# print(jobData)


# calling the function for the ats calculation
# atsmatcher = ATSMatcher(jobData)  # Ensure jobDataFrame is passed correctly
# jobDataFrame = atsmatcher.job_df
# print(jobDataFrame)
# uniqueJobRole=jobDataFrame["Job Role"].unique().tolist()
# uniqueJobRole.append(None)
# print(uniqueJobRole)
# print(structured_data)


@app.route("/ATSscore")
def ats():
    with open("jobs.json", "r") as fp:
        jobData = [json.loads(line) for line in fp]
    atsmatcher=ATSMatcher(jobData)
    jobDataFrame = atsmatcher.job_df
    print(jobDataFrame)
    uniqueJobRole = jobDataFrame["Job Role"].unique().tolist()
    uniqueJobRole.append(None)
    return render_template("ATSScore.html",uniqueJobRole=uniqueJobRole)


@app.route("/uniqueJobRoleFromMERN", methods=["POST"])
def unique_role():
    data = request.get_json()

    # Check if "allJobs" key exists
    if "allJobs" not in data:
        return jsonify({"error": "Missing job data"}), 400

    job_list = data["allJobs"]

    # Convert job list to DataFrame
    jobDataFrame = pd.DataFrame(job_list)

    # Make sure "jobRole" exists in the DataFrame
    if "jobRole" not in jobDataFrame.columns:
        return jsonify({"error": "jobRole column not found"}), 400

    # Get unique job roles
    uniqueJobRoles = jobDataFrame["jobRole"].unique().tolist()

    return jsonify({"uniqueJobRole": uniqueJobRoles,"success": True})



@app.route("/calculate_ats_score", methods=['POST'])
def calculate_ats():
    with open("jobs.json", "r") as fp:
        jobData = [json.loads(line) for line in fp]
    atsmatcher = ATSMatcher(jobData)
    jobDataFrame = atsmatcher.job_df
    print(jobDataFrame)
    opt = request.form.get('selected_option')
    print(opt)

    # Check if resume_details.json exists before trying to open it
    if not os.path.exists("resume_details.json"):
        return jsonify({"error": "resume_details.json not found", "success": False}), 404

    with open("resume_details.json", "r") as f:
        try:
            structured_data = json.load(f)
        except json.JSONDecodeError:
            return jsonify({"error": "Corrupted resume data", "success": False}), 500

    # Extract structured data safely
    cgpa = structured_data["CGPA"] if structured_data["CGPA"] else 0
    exp = structured_data["Work Experience"] if structured_data["Work Experience"] else 0
    skills = structured_data["Skills"] if structured_data["Skills"] else []
    topmatches = atsmatcher.calculate_ats_score(skills, exp, cgpa, opt)
    print(topmatches)

    # Call ATS score function
    if opt == "None":
        matches_list = []
        for _, row in topmatches.iterrows():
            matches_list.append({
                "ID": int(row["ID"]),
                "Job Role": row["Job Role"],
                "ATS Score": float(row["ATS Score"])  # Ensure float format
            })

        return jsonify({"topmatches": matches_list, "success": True})

    if opt:
        filtered_matches = topmatches[topmatches["Job Role"] == opt] if not topmatches.empty else []
        return jsonify({"topmatches": filtered_matches.to_dict(orient="records"), "success": True})


@app.route("/calculate_ats_scoreMERN", methods=['POST'])
def calculate_atsMERN():
    try:
        # Get request data
        data = request.get_json()
        all_jobs = data.get("allJobs", [])
        structured_data = data.get("text", {})
        selected_option = data.get("selectedOption", "None")

        if not all_jobs:
            return jsonify({"error": "Job data is required", "success": False}), 400
        if not structured_data:
            return jsonify({"error": "Structured resume data is required", "success": False}), 400

        # Initialize ATSMatcher
        atsmatcher = ATSMatcher(all_jobs)

        # Extract resume details
        cgpa = float(structured_data.get("CGPA", 0) or 0)
        skills = structured_data.get("Skills", [])

        # Compute ATS score
        top_matches = atsmatcher.calculate_ats_scoreMERN(skills, cgpa, selected_option)

        # Convert result to JSON format

        if selected_option == "None":
            matches_list = top_matches.to_dict(orient="records")
            return jsonify({"topMatches": matches_list, "success": True})
        elif selected_option:
            topMatches=top_matches[top_matches["jobRole"] == selected_option]
            matches_list=topMatches.to_dict(orient='records')
            return jsonify({"topMatches": matches_list, "success": True})




    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


# getting feedback
@app.route("/feedback")
def GetFeedBack():
    if not os.path.exists("resume_details.json"):
        return jsonify({"error": "resume_details.json not found", "success": False}), 404

    with open("resume_details.json", "r") as f:
        try:
            structured_data = json.load(f)
        except json.JSONDecodeError:
            return jsonify({"error": "Corrupted resume data", "success": False}), 500
    data=get_resume_feedback(structured_data)
    if data :
        if os.path.exists("resume_details.json"):
            os.remove("resume_details.json")
            print("resume_details.json deleted successfully.")
        return jsonify({"success" : True,"data": data}) , 200
    else:
        if os.path.exists("resume_details.json"):
            os.remove("resume_details.json")
            print("resume_details.json deleted successfully.")
        return jsonify({"success" : False,"message" : "No Feedback is Present"}), 500


@app.route("/feedbackMERN", methods=["POST"])
def GetFeedBackMERN():
    # Assuming the structured data is being passed as a JSON payload in the request body
    data1 = request.get_json()
    structured_data = data1.get("text", {})

    if not structured_data:
        return jsonify({"error": "No structured data provided", "success": False}), 400

    data = get_resume_feedback(structured_data)

    if data:
        # Assuming that the feedback will be returned with the structured data
        return jsonify({"success": True, "data": data}), 200
    else:
        return jsonify({"success": False, "message": "No feedback is present"}), 500



if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
