import streamlit as st
import PyPDF2
import json
import requests
from io import BytesIO
import re
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Constants
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def sanitize_text(text):
    """Clean and sanitize extracted text."""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,@()-]', '', text)
    return text.strip()

def create_prompt(text):
    """Create a well-structured prompt for the Mistral API."""
    return {
        "model": "mistral-small-latest",
        "messages": [
            {
                "role": "system",
                "content": """You are a professional resume parser and skills classifier. Your task is to extract information from the resume and return it in JSON format. 

IMPORTANT: Return ONLY the raw JSON object without any markdown formatting, code blocks, or additional text.

For skills classification, categorize skills into the following groups:
- Technical Skills: Programming languages, frameworks, technical methodologies
- Tools & Software: Specific software applications, platforms, tools
- Soft Skills: Communication, leadership, interpersonal skills
- Domain Knowledge: Industry-specific knowledge, methodologies, processes
- Certifications & Training: Professional certifications, specialized training
- Management Skills: Project management, team management, resource planning
- Analytics & Data: Data analysis, reporting, business intelligence
- Languages: Programming languages should go under Technical Skills, human languages here

Extract and return the following fields in JSON:
{
    "full_name": "string",
    "email": "string",
    "phone": "string",
    "education": [{
        "degree": "string",
        "institution": "string",
        "year": "string",
        "field_of_study": "string"
    }],
    "work_experience": [{
        "position": "string",
        "company": "string",
        "dates": "string",
        "responsibilities": ["string"],
        "achievements": ["string"]
    }],
    "skills": {
        "technical_skills": ["string"],
        "tools_and_software": ["string"],
        "soft_skills": ["string"],
        "domain_knowledge": ["string"],
        "certifications_and_training": ["string"],
        "management_skills": ["string"],
        "analytics_and_data": ["string"]
    },
    "languages": [{
        "language": "string",
        "proficiency": "string"
    }],
    "certifications": [{
        "name": "string",
        "issuer": "string",
        "date": "string"
    }]
}

Ensure all skills are properly categorized based on the classification above. If a category has no skills, include an empty array."""
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "temperature": 0.3,
        "max_tokens": 1500
    }

def clean_json_response(response_text):
    """Clean the API response to get valid JSON."""
    # Remove any markdown code blocks
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*', '', response_text)
    
    # Remove any leading/trailing whitespace
    response_text = response_text.strip()
    
    return response_text

def call_mistral_api(prompt):
    """Make API call to Mistral."""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            MISTRAL_API_URL,
            headers=headers,
            json=prompt,
            timeout=30
        )
        
        response.raise_for_status()
        
        content = response.json()["choices"][0]["message"]["content"]
        cleaned_content = clean_json_response(content)
        
        # Validate JSON
        return json.loads(cleaned_content)
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"JSON Parsing Error: {str(e)}")
        st.text("Raw Response:")
        st.code(content)
        return None

def display_skills_section(skills):
    """Display skills section with expandable categories."""
    if not skills:
        return
    
    st.subheader("🛠️ Skills Analysis")
    
    # Define skill categories with emojis
    skill_categories = {
        "technical_skills": "💻 Technical Skills",
        "tools_and_software": "🔧 Tools & Software",
        "soft_skills": "🤝 Soft Skills",
        "domain_knowledge": "🎯 Domain Knowledge",
        "certifications_and_training": "📜 Certifications & Training",
        "management_skills": "👥 Management Skills",
        "analytics_and_data": "📊 Analytics & Data"
    }
    
    # Create columns for skills display
    cols = st.columns(2)
    col_idx = 0
    
    for key, title in skill_categories.items():
        if skills.get(key):
            with cols[col_idx]:
                with st.expander(title):
                    for skill in skills[key]:
                        st.write(f"- {skill}")
            col_idx = (col_idx + 1) % 2

def display_resume_info(resume_data):
    """Display parsed resume information in a structured format."""
    if not resume_data:
        return
    
    st.header("📄 Resume Information")
    
    # Personal Information
    st.subheader("👤 Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Name:** {resume_data.get('full_name', 'N/A')}")
    with col2:
        st.write(f"**Email:** {resume_data.get('email', 'N/A')}")
    st.write(f"**Phone:** {resume_data.get('phone', 'N/A')}")
    
    # Education
    if resume_data.get('education'):
        st.subheader("🎓 Education")
        for edu in resume_data['education']:
            st.write(f"- **{edu.get('degree', '')}** in {edu.get('field_of_study', '')}")
            st.write(f"  {edu.get('institution', '')}, {edu.get('year', '')}")
    
    # Work Experience
    if resume_data.get('work_experience'):
        st.subheader("💼 Work Experience")
        for exp in resume_data['work_experience']:
            st.write(f"**{exp.get('position', '')} at {exp.get('company', '')}**")
            st.write(f"*{exp.get('dates', '')}*")
            
            if exp.get('responsibilities'):
                with st.expander("Responsibilities"):
                    for resp in exp['responsibilities']:
                        st.write(f"- {resp}")
            
            if exp.get('achievements'):
                with st.expander("Achievements"):
                    for achievement in exp['achievements']:
                        st.write(f"- {achievement}")
    
    # Skills (using the new categorized display)
    if resume_data.get('skills'):
        display_skills_section(resume_data['skills'])
    
    # Languages
    if resume_data.get('languages'):
        st.subheader("🌐 Languages")
        for lang in resume_data['languages']:
            st.write(f"- **{lang.get('language', '')}**: {lang.get('proficiency', '')}")
    
    # Certifications
    if resume_data.get('certifications'):
        st.subheader("📜 Certifications")
        for cert in resume_data['certifications']:
            st.write(f"- **{cert.get('name', '')}**")
            st.write(f"  Issued by {cert.get('issuer', '')} ({cert.get('date', '')})")

def main():
    st.set_page_config(page_title="Resume Parser", page_icon="📄", layout="wide")
    
    st.title("📄 Advanced Resume Parser")
    st.write("Upload a resume PDF to extract and analyze its contents with detailed skills classification.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing resume..."):
            # Extract text from PDF
            text = extract_text_from_pdf(uploaded_file)
            
            # Sanitize text
            cleaned_text = sanitize_text(text)
            
            # Create prompt and call Mistral API
            prompt = create_prompt(cleaned_text)
            resume_info = call_mistral_api(prompt)
            
            if resume_info:
                # Display parsed information
                display_resume_info(resume_info)
                
                # Show raw JSON in expander
                with st.expander("View Raw JSON"):
                    st.json(resume_info)

if __name__ == "__main__":
    main()