import streamlit as st
import os
import pandas as pd
import sqlite3
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import PyPDF2
import json
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# API keys for LLM services
groq_api_key = os.getenv("GROQ_API_KEY")
email_password = os.getenv("EMAIL_PASSWORD")
sender_email = os.getenv("SENDER_EMAIL")

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
    from langchain_groq import ChatGroq
    from langchain.chains.summarize import load_summarize_chain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
else:
    st.error("GROQ_API_KEY is missing. Please set it in the .env file.")

# Initialize database
def init_db():
    conn = sqlite3.connect("job_screening.db")
    c = conn.cursor()
    # Job Descriptions table
    c.execute("""
    CREATE TABLE IF NOT EXISTS job_descriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_title TEXT,
        job_description TEXT,
        required_skills TEXT,
        experience_required TEXT,
        qualifications TEXT,
        responsibilities TEXT,
        summary TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Candidates table
    c.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        education TEXT,
        work_experience TEXT,
        skills TEXT,
        certifications TEXT,
        resume_text TEXT,
        resume_file_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Matches table
    c.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER,
        candidate_id INTEGER,
        match_score REAL,
        shortlisted BOOLEAN DEFAULT 0,
        interview_scheduled BOOLEAN DEFAULT 0,
        interview_date TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (job_id) REFERENCES job_descriptions (id),
        FOREIGN KEY (candidate_id) REFERENCES candidates (id)
    )
    """)
    
    conn.commit()
    conn.close()

# Agent 1: Job Description Summarizer
class JobDescriptionSummarizer:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
        
    def summarize(self, job_description):
        # Split text for processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=400
        )
        docs = [Document(page_content=job_description)]
        
        # Create prompt template
        prompt_template = """
        Extract and summarize the following key elements from this job description:
        
        Job Description: {text}
        
        Please provide a structured output in JSON format with these fields:
        1. Required skills (list)
        2. Experience required (years and type)
        3. Qualifications (education, certifications)
        4. Job responsibilities (list)
        5. Brief summary (2-3 sentences)
        
        Format as valid JSON.
        """
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        
        chain = load_summarize_chain(
            self.llm,
            chain_type="stuff",
            prompt=PROMPT
        )
        
        summary = chain.run(docs)
        try:
            # Clean the response to ensure it's valid JSON
            json_start = summary.find('{')
            json_end = summary.rfind('}') + 1
            clean_json = summary[json_start:json_end]
            result = json.loads(clean_json)
            return result
        except Exception as e:
            st.error(f"Error parsing summarization result: {e}")
            return {
                "required_skills": [],
                "experience_required": "",
                "qualifications": "",
                "responsibilities": [],
                "summary": "Failed to parse summary"
            }

# Agent 2: Recruiting Agent (CV Parser)
class RecruitingAgent:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    
    def extract_text_from_pdf(self, pdf_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.read())
            temp_path = temp_file.name
        
        text = ""
        try:
            with open(temp_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
        finally:
            os.unlink(temp_path)
        
        return text
    
    def extract_cv_data(self, cv_text):
        prompt_template = """
        Extract the following information from this resume/CV:
        
        Resume Text: {text}
        
        Please provide a structured output in JSON format with these fields:
        1. Name
        2. Email
        3. Education (list of degrees, institutions)
        4. Work Experience (list of roles, companies, durations)
        5. Skills (list of technical and soft skills)
        6. Certifications (list)
        
        Format as valid JSON.
        """
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        
        try:
            response = self.llm.predict(PROMPT.format(text=cv_text[:10000]))  # Limit text length
            # Clean the response to ensure it's valid JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            clean_json = response[json_start:json_end]
            result = json.loads(clean_json)
            return result
        except Exception as e:
            st.error(f"Error extracting CV data: {e}")
            return {
                "name": "Unknown",
                "email": "unknown@example.com",
                "education": [],
                "work_experience": [],
                "skills": [],
                "certifications": []
            }
    
    def calculate_match_score(self, jd_summary, cv_data):
        # Convert lists to strings for processing
        jd_skills = " ".join(jd_summary.get("required_skills", []))
        cv_skills = " ".join(cv_data.get("skills", []))
        
        # Calculate skills similarity
        skills_similarity = 0.0
        if jd_skills.strip() and cv_skills.strip():
            # Only use TF-IDF if both have non-empty skills
            try:
                # Create vectors
                vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
                vectors = vectorizer.fit_transform([jd_skills, cv_skills])
                
                # Calculate similarity between skills
                skills_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            except Exception as e:
                st.warning(f"Error in skills comparison: {e}")
                # Fallback: Count matching skills
                jd_skill_list = set([s.lower().strip() for s in jd_summary.get("required_skills", [])])
                cv_skill_list = set([s.lower().strip() for s in cv_data.get("skills", [])])
                
                if jd_skill_list and cv_skill_list:
                    matching_skills = jd_skill_list.intersection(cv_skill_list)
                    skills_similarity = len(matching_skills) / max(len(jd_skill_list), 1)
        
        # Calculate experience match (basic implementation)
        exp_required = jd_summary.get("experience_required", "")
        exp_match = 0.5  # Default medium match
        if exp_required:
            try:
                # Try to extract years from experience required
                years_required = re.search(r'(\d+)', exp_required)
                if years_required:
                    years_required = int(years_required.group(1))
                    # Estimate candidate's experience from work history
                    candidate_exp_years = len(cv_data.get("work_experience", []))
                    exp_diff = abs(candidate_exp_years - years_required)
                    if exp_diff <= 1:
                        exp_match = 1.0  # Close match
                    elif exp_diff <= 3:
                        exp_match = 0.7  # Good match
                    else:
                        exp_match = 0.3  # Poor match
            except:
                pass
        
        # Calculate education/qualification match
        qual_match = 0.5  # Default medium match
        jd_qualifications = jd_summary.get("qualifications", "").lower()
        cv_education = " ".join([str(edu) for edu in cv_data.get("education", [])]).lower()
        
        edu_keywords = ["bachelor", "master", "phd", "degree", "diploma", "certificate"]
        matches = 0
        for keyword in edu_keywords:
            if keyword in jd_qualifications and keyword in cv_education:
                matches += 1
        
        if matches > 0:
            qual_match = min(1.0, matches / 3)
        
        # Combine scores with weights
        weights = {"skills": 0.5, "experience": 0.3, "qualifications": 0.2}
        final_score = (
            weights["skills"] * skills_similarity + 
            weights["experience"] * exp_match + 
            weights["qualifications"] * qual_match
        )
        
        return min(1.0, max(0.0, final_score))  # Ensure score is between 0 and 1

# Agent 3: Shortlisting Candidates
class CandidateShortlister:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
    
    def shortlist(self, match_scores):
        shortlisted = []
        for candidate_id, score in match_scores.items():
            if score >= self.threshold:
                shortlisted.append((candidate_id, score))
        
        # Sort by score in descending order
        shortlisted.sort(key=lambda x: x[1], reverse=True)
        return shortlisted

# Agent 4: Interview Scheduler
class InterviewScheduler:
    def __init__(self, sender_email, email_password):
        self.sender_email = sender_email
        self.email_password = email_password
    
    def generate_interview_slots(self, num_slots=3):
        # Generate interview slots starting from tomorrow
        start_date = datetime.now() + timedelta(days=1)
        slots = []
        
        for i in range(num_slots):
            slot_date = start_date + timedelta(days=i)
            # Create morning and afternoon slots
            morning_slot = datetime.combine(slot_date.date(), datetime.strptime("10:00", "%H:%M").time())
            afternoon_slot = datetime.combine(slot_date.date(), datetime.strptime("14:00", "%H:%M").time())
            slots.extend([morning_slot, afternoon_slot])
        
        return slots
    
    def send_interview_request(self, candidate_data, job_data, slots):
        # Create email content
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = candidate_data['email']
        msg['Subject'] = f"Interview Request: {job_data['job_title']} Position"
        
        # Email body
        body = f"""
        Dear {candidate_data['name']},
        
        We are pleased to inform you that your application for the {job_data['job_title']} position has been shortlisted.
        
        We would like to invite you for an interview. Please select one of the following time slots:
        
        """
        
        # Add slots
        for i, slot in enumerate(slots):
            body += f"{i+1}. {slot.strftime('%A, %B %d, %Y at %I:%M %p')}\n"
        
        body += """
        Please reply to this email with your preferred slot number.
        
        We look forward to speaking with you.
        
        Best regards,
        Recruitment Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.sender_email, self.email_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, candidate_data['email'], text)
            server.quit()
            return True, "Email sent successfully"
        except Exception as e:
            return False, str(e)

# Main Application
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    return temp_path

def load_job_descriptions():
    conn = sqlite3.connect("job_screening.db")
    df = pd.read_sql_query("SELECT * FROM job_descriptions", conn)
    conn.close()
    return df

def load_candidates():
    conn = sqlite3.connect("job_screening.db")
    df = pd.read_sql_query("SELECT * FROM candidates", conn)
    conn.close()
    return df

def load_matches():
    conn = sqlite3.connect("job_screening.db")
    query = """
    SELECT m.id, m.job_id, m.candidate_id, m.match_score, m.shortlisted, m.interview_scheduled,
           j.job_title, c.name, c.email
    FROM matches m
    JOIN job_descriptions j ON m.job_id = j.id
    JOIN candidates c ON m.candidate_id = c.id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def save_job_description(title, description, summary):
    conn = sqlite3.connect("job_screening.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO job_descriptions (job_title, job_description, required_skills, experience_required, qualifications, responsibilities, summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            title,
            description,
            json.dumps(summary.get("required_skills", [])),
            summary.get("experience_required", ""),
            summary.get("qualifications", ""),
            json.dumps(summary.get("responsibilities", [])),
            summary.get("summary", "")
        )
    )
    job_id = c.lastrowid
    conn.commit()
    conn.close()
    return job_id

def save_candidate(cv_data, resume_text, file_path):
    conn = sqlite3.connect("job_screening.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO candidates (name, email, education, work_experience, skills, certifications, resume_text, resume_file_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            cv_data.get("name", "Unknown"),
            cv_data.get("email", "unknown@example.com"),
            json.dumps(cv_data.get("education", [])),
            json.dumps(cv_data.get("work_experience", [])),
            json.dumps(cv_data.get("skills", [])),
            json.dumps(cv_data.get("certifications", [])),
            resume_text,
            file_path
        )
    )
    candidate_id = c.lastrowid
    conn.commit()
    conn.close()
    return candidate_id

def save_match(job_id, candidate_id, match_score):
    conn = sqlite3.connect("job_screening.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO matches (job_id, candidate_id, match_score) VALUES (?, ?, ?)",
        (job_id, candidate_id, match_score)
    )
    match_id = c.lastrowid
    conn.commit()
    conn.close()
    return match_id

def update_shortlisted(match_id, shortlisted=True):
    conn = sqlite3.connect("job_screening.db")
    c = conn.cursor()
    c.execute(
        "UPDATE matches SET shortlisted = ? WHERE id = ?",
        (1 if shortlisted else 0, match_id)
    )
    conn.commit()
    conn.close()

def update_interview_scheduled(match_id, scheduled=True, date=None):
    conn = sqlite3.connect("job_screening.db")
    c = conn.cursor()
    c.execute(
        "UPDATE matches SET interview_scheduled = ?, interview_date = ? WHERE id = ?",
        (1 if scheduled else 0, date, match_id)
    )
    conn.commit()
    conn.close()

def main():
    st.set_page_config(page_title="AI-Powered Job Screening System", layout="wide")
    
    # Initialize database
    init_db()
    
    # Sidebar for navigation 
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Job Description", "Upload Resumes", "Match & Shortlist", "Schedule Interviews", "Dashboard"])
    
    if page == "Upload Job Description":
        st.title("Upload Job Description")
        
        job_title = st.text_input("Job Title")
        job_description = st.text_area("Job Description", height=300)
        
        if st.button("Summarize Job Description"):
            if job_title and job_description:
                with st.spinner("Summarizing job description..."):
                    # Agent 1: Summarize JD
                    summarizer = JobDescriptionSummarizer()
                    summary = summarizer.summarize(job_description)
                    
                    # Display summary
                    st.subheader("Job Description Summary")
                    st.write("**Required Skills:**")
                    st.write(", ".join(summary.get("required_skills", [])))
                    
                    st.write("**Experience Required:**")
                    st.write(summary.get("experience_required", ""))
                    
                    st.write("**Qualifications:**")
                    st.write(summary.get("qualifications", ""))
                    
                    st.write("**Responsibilities:**")
                    for resp in summary.get("responsibilities", []):
                        st.write(f"- {resp}")
                    
                    st.write("**Summary:**")
                    st.write(summary.get("summary", ""))
                    
                    # Save to database
                    job_id = save_job_description(job_title, job_description, summary)
                    st.success(f"Job description saved with ID: {job_id}")
            else:
                st.warning("Please enter both job title and description")

    elif page == "Upload Resumes":
        st.title("Upload Candidate Resumes")
        
        uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Process Resumes"):
            if uploaded_files:
                recruiting_agent = RecruitingAgent()
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        # Extract text from PDF
                        resume_text = recruiting_agent.extract_text_from_pdf(uploaded_file)
                        
                        # Extract data from CV
                        cv_data = recruiting_agent.extract_cv_data(resume_text)
                        
                        # Save to file system
                        file_path = save_uploaded_file(uploaded_file)
                        
                        # Save to database
                        candidate_id = save_candidate(cv_data, resume_text, file_path)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.success(f"Processed {len(uploaded_files)} resumes")
            else:
                st.warning("Please upload resume files")
    
    elif page == "Match & Shortlist":
        st.title("Match Candidates to Job Descriptions")
        
        # Load job descriptions and candidates
        job_df = load_job_descriptions()
        candidate_df = load_candidates()
        
        if len(job_df) == 0:
            st.warning("No job descriptions available. Please upload a job description first.")
        elif len(candidate_df) == 0:
            st.warning("No candidates available. Please upload resumes first.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Select Job Description")
                job_options = job_df["job_title"].tolist()
                selected_job = st.selectbox("Job", job_options)
                selected_job_id = job_df[job_df["job_title"] == selected_job]["id"].iloc[0]
                
                # Display job summary
                job_row = job_df[job_df["id"] == selected_job_id].iloc[0]
                st.write("**Required Skills:**")
                required_skills = json.loads(job_row["required_skills"])
                if required_skills:
                    st.write(", ".join(required_skills))
                else:
                    st.write("No specific skills listed")
                    
                st.write("**Experience Required:**")
                st.write(job_row["experience_required"] or "Not specified")
            
            with col2:
                st.subheader("Matching Threshold")
                threshold = st.slider("Set minimum match score for shortlisting", min_value=0.5, max_value=0.95, value=0.8, step=0.05)
            
            if st.button("Match Candidates"):
                with st.spinner("Matching candidates..."):
                    recruiting_agent = RecruitingAgent()
                    shortlister = CandidateShortlister(threshold=threshold)
                    
                    # Get job summary
                    job_row = job_df[job_df["id"] == selected_job_id].iloc[0]
                    job_summary = {
                        "required_skills": json.loads(job_row["required_skills"]),
                        "experience_required": job_row["experience_required"],
                        "qualifications": job_row["qualifications"],
                        "responsibilities": json.loads(job_row["responsibilities"]),
                        "summary": job_row["summary"]
                    }
                    
                    # Process each candidate
                    match_scores = {}
                    progress_bar = st.progress(0)
                    
                    for i, (_, candidate) in enumerate(candidate_df.iterrows()):
                        # Prepare CV data
                        cv_data = {
                            "name": candidate["name"],
                            "email": candidate["email"],
                            "education": json.loads(candidate["education"]),
                            "work_experience": json.loads(candidate["work_experience"]),
                            "skills": json.loads(candidate["skills"]),
                            "certifications": json.loads(candidate["certifications"])
                        }
                        
                        # Calculate match score
                        match_score = recruiting_agent.calculate_match_score(job_summary, cv_data)
                        match_scores[candidate["id"]] = match_score
                        
                        # Save match to database
                        save_match(selected_job_id, candidate["id"], match_score)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(candidate_df))
                    
                    # Shortlist candidates
                    shortlisted = shortlister.shortlist(match_scores)
                    
                    # Update database for shortlisted candidates
                    for candidate_id, score in shortlisted:
                        conn = sqlite3.connect("job_screening.db")
                        c = conn.cursor()
                        c.execute(
                            "UPDATE matches SET shortlisted = 1 WHERE job_id = ? AND candidate_id = ?",
                            (selected_job_id, candidate_id)
                        )
                        conn.commit()
                        conn.close()
                    
                    # Display results
                    st.subheader("Matching Results")
                    results = []
                    
                    for candidate_id, score in match_scores.items():
                        candidate_row = candidate_df[candidate_df["id"] == candidate_id].iloc[0]
                        results.append({
                            "ID": candidate_id,
                            "Name": candidate_row["name"],
                            "Email": candidate_row["email"],
                            "Match Score": f"{score:.2f}",
                            "Shortlisted": "Yes" if score >= threshold else "No"
                        })
                    
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values("Match Score", ascending=False)
                    st.dataframe(results_df)
                    
                    st.success(f"Shortlisted {len(shortlisted)} candidates out of {len(candidate_df)} applications")
    
    elif page == "Schedule Interviews":
        st.title("Schedule Interviews")
        
        # Load shortlisted candidates
        matches_df = load_matches()
        shortlisted_df = matches_df[matches_df["shortlisted"] == 1]
        
        if len(shortlisted_df) == 0:
            st.warning("No shortlisted candidates available. Please match and shortlist candidates first.")
        else:
            # Filter by job
            job_options = shortlisted_df["job_title"].unique().tolist()
            selected_job = st.selectbox("Select Job", job_options)
            filtered_df = shortlisted_df[shortlisted_df["job_title"] == selected_job]
            
            st.subheader("Shortlisted Candidates")
            st.dataframe(filtered_df[["name", "email", "match_score", "interview_scheduled"]])
            
            # Select candidate for scheduling
            candidate_options = filtered_df["name"].tolist()
            selected_candidate = st.selectbox("Select Candidate for Interview", candidate_options)
            
            # Email settings
            if sender_email and email_password:
                if st.button("Schedule Interview"):
                    # Get candidate data
                    candidate_row = filtered_df[filtered_df["name"] == selected_candidate].iloc[0]
                    
                    # Generate interview slots
                    scheduler = InterviewScheduler(sender_email, email_password)
                    slots = scheduler.generate_interview_slots()
                    
                    # Format slots for display
                    slot_texts = [slot.strftime("%A, %B %d, %Y at %I:%M %p") for slot in slots]
                    st.write("**Interview Slots Generated:**")
                    for slot in slot_texts:
                        st.write(f"- {slot}")
                    
                    # Send email
                    candidate_data = {
                        "name": candidate_row["name"],
                        "email": candidate_row["email"]
                    }
                    
                    job_data = {
                        "job_title": candidate_row["job_title"]
                    }
                    
                    success, message = scheduler.send_interview_request(candidate_data, job_data, slots)
                    
                    if success:
                        # Update database
                        update_interview_scheduled(candidate_row["id"], True, slots[0].strftime("%Y-%m-%d %H:%M:%S"))
                        st.success(f"Interview request sent to {candidate_row['name']} ({candidate_row['email']})")
                    else:
                        st.error(f"Failed to send email: {message}")
            else:
                st.warning("Email credentials not configured. Please set SENDER_EMAIL and EMAIL_PASSWORD in the .env file to enable interview scheduling.")
    
    elif page == "Dashboard":
        st.title("Recruitment Dashboard")
        
        # Load data
        job_df = load_job_descriptions()
        candidate_df = load_candidates()
        matches_df = load_matches()
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Jobs", len(job_df))
        
        with col2:
            st.metric("Total Candidates", len(candidate_df))
        
        with col3:
            shortlisted_count = len(matches_df[matches_df["shortlisted"] == 1])
            st.metric("Shortlisted", shortlisted_count)
        
        with col4:
            interviews_count = len(matches_df[matches_df["interview_scheduled"] == 1])
            st.metric("Interviews Scheduled", interviews_count)
        
        # Jobs table
        st.subheader("Job Listings")
        if len(job_df) > 0:
            job_display = job_df[["job_title", "summary", "created_at"]]
            job_display = job_display.rename(columns={"created_at": "Posted Date"})
            st.dataframe(job_display)
        else:
            st.info("No job listings available")
        
        # Candidate matching stats
        st.subheader("Candidate Match Distribution")
        if len(matches_df) > 0:
            # Create histogram of match scores
            hist_data = matches_df["match_score"].tolist()
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.hist(hist_data, bins=10, alpha=0.7)
            ax.set_xlabel("Match Score")
            ax.set_ylabel("Number of Candidates")
            ax.set_title("Distribution of Candidate Match Scores")
            st.pyplot(fig)
        else:
            st.info("No matching data available")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()