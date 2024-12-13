import streamlit as st
import fitz  # PyMuPDF
import io
import json
import base64
import google.generativeai as genai
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from docx import Document

# Configure the API key
genai.configure(api_key=st.secrets.GOOGLE_API_KEY)

# Define cached functions
@st.cache_data()
def get_gemini_response(input, pdf_content, prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Add specific instructions for match score format
        enhanced_prompt = prompt + "\n\nIMPORTANT: Your response MUST include a 'Match Score: X%' line where X is a number between 0 and 100 based on the candidate's fit for the role."
        response = model.generate_content([
            "Job Description:\n" + input + "\n\nResume Content:\n" + pdf_content + "\n\n" + enhanced_prompt
        ])
        return response.text
    except Exception as e:
        st.error(f"Error in API response: {e}")
        return "Error generating response"

@st.cache_data()
def get_gemini_response_keywords(input, pdf_content, prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([
            "Based on this job description:\n" + input + "\n\nAnd this resume:\n" + pdf_content + "\n\n" + prompt
        ])
        response_text = response.text.strip()
        
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'(\{[\s\S]*\})', response_text)
        if json_match:
            try:
                skills_dict = json.loads(json_match.group(1))
                return json.dumps(skills_dict)  # Convert back to JSON string
            except json.JSONDecodeError:
                pass
        
        # If no valid JSON found, create a structured response
        skills = {
            "Technical Skills": [],
            "Analytical Skills": [],
            "Soft Skills": []
        }
        
        # Extract skills using regex patterns
        tech_skills = re.findall(r'Technical Skills?:?\s*[-:]*\s*((?:[^:\n]+(?:,|and|\n|$))+)', response_text, re.IGNORECASE)
        analytical_skills = re.findall(r'Analytical Skills?:?\s*[-:]*\s*((?:[^:\n]+(?:,|and|\n|$))+)', response_text, re.IGNORECASE)
        soft_skills = re.findall(r'Soft Skills?:?\s*[-:]*\s*((?:[^:\n]+(?:,|and|\n|$))+)', response_text, re.IGNORECASE)
        
        # Process found skills
        if tech_skills:
            skills["Technical Skills"] = [s.strip() for s in re.split(r'[,\n]|and', tech_skills[0]) if s.strip()]
        if analytical_skills:
            skills["Analytical Skills"] = [s.strip() for s in re.split(r'[,\n]|and', analytical_skills[0]) if s.strip()]
        if soft_skills:
            skills["Soft Skills"] = [s.strip() for s in re.split(r'[,\n]|and', soft_skills[0]) if s.strip()]
        
        return json.dumps(skills)  # Ensure we return a JSON string
    except Exception as e:
        st.error(f"Error in API response: {e}")
        return json.dumps({
            "Technical Skills": [],
            "Analytical Skills": [],
            "Soft Skills": []
        })

@st.cache_data()
def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        # Read PDF file
        pdf_bytes = uploaded_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Get first page
        first_page = pdf_document[0]
        
        # Convert to image with RGB colorspace
        pix = first_page.get_pixmap(alpha=False)
        
        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        pdf_document.close()
        
        pdf_parts = base64.b64encode(img_byte_arr).decode()
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

@st.cache_data()
def input_docx_setup(uploaded_file):
    if uploaded_file is not None:
        doc = Document(uploaded_file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        text = '\n'.join(full_text)
        return text
    else:
        raise FileNotFoundError("No file uploaded")

def analyze_resume(resume_content, job_description):
    input_prompt = """
    You are an advanced ATS with a sophisticated matching algorithm. Consider the following aspects:
    1. Skills Weight System: Differentiate between must-have and nice-to-have skills.
    2. Experience Level Matching: Consider years of experience in skill matching.
    3. Education Requirements Matching: Compare candidate qualifications with job requirements.
    4. Project Experience Analysis: Match project experiences with job requirements.
    Provide a comprehensive analysis.
    """
    response = get_gemini_response(job_description, resume_content, input_prompt)
    return response

def visualize_skills_gap(keywords):
    # Convert the skills dictionary into a format suitable for plotting
    data = []
    for skill_type, skills_list in {
        "Technical Skills": keywords.get('Technical Skills', []),
        "Analytical Skills": keywords.get('Analytical Skills', []),
        "Soft Skills": keywords.get('Soft Skills', []),
    }.items():
        for skill in skills_list:
            data.append({
                'Skill Type': skill_type,
                'Skill': skill,
                'Count': 1  # Each skill counts as 1
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    if df.empty:
        st.warning("No skills data available to visualize")
        return
    
    # Create grouped bar chart
    fig = px.bar(df, 
                 x='Skill Type',
                 y='Count',
                 title="Skills Distribution",
                 color='Skill Type',
                 text=df.groupby('Skill Type')['Count'].transform('count'))
    
    # Update layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Skill Categories",
        yaxis_title="Number of Skills",
        showlegend=False
    )
    
    st.plotly_chart(fig)

def calculate_match_score(response_text):
    try:
        if not response_text or "Match Score:" not in response_text:
            return 0
        # Find the match score using regex
        import re
        match = re.search(r'Match Score:\s*(\d+(?:\.\d+)?)\s*%', response_text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 0
    except Exception as e:
        st.error(f"Error extracting match score: {e}")
        return 0

def visualize_match_score(match_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=match_score,
        title={'text': "Resume Match Score"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 50], 'color': "red"},
                   {'range': [50, 75], 'color': "yellow"},
                   {'range': [75, 100], 'color': "green"}]}))
    st.plotly_chart(fig)

def batch_process_resumes(files, job_description):
    results = []
    for file in files:
        try:
            if file.type == "application/pdf":
                resume_content = input_pdf_setup(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_content = input_docx_setup(file)
            else:
                continue

            # Get detailed analysis for the resume
            response = get_gemini_response(job_description, resume_content, input_prompt3)
            
            # Extract match score
            match_score = calculate_match_score(response)
            
            # Extract key points from response
            import re
            
            # Extract detailed assessment
            detailed_assessment = ""
            assessment_match = re.search(r'Detailed Assessment:(.*?)(?=Key Strengths:|$)', response, re.DOTALL)
            if assessment_match:
                detailed_assessment = assessment_match.group(1).strip()
            
            # Extract key strengths
            strengths = []
            strengths_match = re.search(r'Key Strengths:(.*?)(?=Areas for Development:|$)', response, re.DOTALL)
            if strengths_match:
                strengths = [s.strip('- ').strip() for s in strengths_match.group(1).strip().split('\n') if s.strip('- ').strip()]
            
            # Extract areas for development
            areas_for_development = []
            development_match = re.search(r'Areas for Development:(.*?)(?=$)', response, re.DOTALL)
            if development_match:
                areas_for_development = [s.strip('- ').strip() for s in development_match.group(1).strip().split('\n') if s.strip('- ').strip()]
            
            # Create structured result
            result = {
                'File Name': file.name,
                'Match Score': f"{match_score}%",
                'Detailed Assessment': detailed_assessment,
                'Key Strengths': ', '.join(strengths) if strengths else 'None specified',
                'Areas for Development': ', '.join(areas_for_development) if areas_for_development else 'None specified'
            }
            
            results.append(result)
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            results.append({
                'File Name': file.name,
                'Match Score': "Error",
                'Detailed Assessment': f"Error processing file: {str(e)}",
                'Key Strengths': 'Error',
                'Areas for Development': 'Error'
            })
    
    return results

# Streamlit App
st.set_page_config(page_title="ATS Resume Scanner")
st.header("🚀 Next-Gen Applicant Tracking System")
input_text = st.text_area("Job Description: ", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF/DOCX)...", type=["pdf", "docx"])
uploaded_files = st.file_uploader("Upload multiple resumes for batch processing (PDF/DOCX)...", type=["pdf", "docx"], accept_multiple_files=True)

if 'resume' not in st.session_state:
    st.session_state.resume = None

if uploaded_file is not None:
    st.write("File Uploaded Successfully")
    if uploaded_file.type == "application/pdf":
        st.session_state.resume = input_pdf_setup(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        st.session_state.resume = input_docx_setup(uploaded_file)

col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    submit1 = st.button("Tell Me About the Resume")

with col2:
    submit2 = st.button("Get Keywords")

with col3:
    submit3 = st.button("Percentage Match")

with col4:
    batch_process = st.button("Batch Process Resumes")

input_prompt2 = """
Extract and categorize skills from the resume into the following JSON format:
{
    "Technical Skills": ["skill1", "skill2", ...],
    "Analytical Skills": ["skill1", "skill2", ...],
    "Soft Skills": ["skill1", "skill2", ...]
}
Ensure to extract ALL relevant skills from the resume text. Include programming languages, tools, frameworks, methodologies, and any other relevant skills.
"""

input_prompt3 = """
You are an advanced ATS (Applicant Tracking System). Analyze the resume against the job description and provide:

1. A detailed assessment of the candidate's fit for the role
2. A percentage match score based on:
   - Skills alignment (50%)
   - Experience relevance (30%)
   - Education fit (20%)

Format your response as:
Match Score: X%

Detailed Assessment:
[Your detailed assessment here]

Key Strengths:
- [List key strengths]

Areas for Development:
- [List areas for improvement]
"""

if submit1:
    if st.session_state.resume is not None:
        response = analyze_resume(st.session_state.resume, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.write("Please upload the resume")

elif submit2:
    if st.session_state.resume is not None:
        response = get_gemini_response_keywords(input_text, st.session_state.resume, input_prompt2)
        st.subheader("Skills are:")
        if response is not None:
            try:
                keywords = json.loads(response)
                st.write(f"Technical Skills: {', '.join(keywords['Technical Skills'])}.")
                st.write(f"Analytical Skills: {', '.join(keywords['Analytical Skills'])}.")
                st.write(f"Soft Skills: {', '.join(keywords['Soft Skills'])}.")
                visualize_skills_gap(keywords)
            except json.JSONDecodeError as e:
                st.error(f"Error decoding JSON: {e}")
                st.write("Response received:")
                st.write(response)
        else:
            st.write("No response received from the API")
    else:
        st.write("Please upload the resume")

elif submit3:
    if st.session_state.resume is not None:
        response = get_gemini_response(input_text, st.session_state.resume, input_prompt3)
        match_score = calculate_match_score(response)
        st.subheader("The Response is")
        st.write(response)
        visualize_match_score(match_score)
    else:
        st.write("Please upload the resume")

elif batch_process:
    if uploaded_files:
        results = batch_process_resumes(uploaded_files, input_text)
        st.subheader("Batch Processing Results")
        for result in results:
            st.write(f"Resume: {result['File Name']}")
            st.write(f"Match Score: {result['Match Score']}")
            st.write(f"Detailed Assessment: {result['Detailed Assessment']}")
            st.write(f"Key Strengths: {result['Key Strengths']}")
            st.write(f"Areas for Development: {result['Areas for Development']}")
        df = pd.DataFrame(results)
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='batch_processing_results.csv',
            mime='text/csv',
        )
        with io.BytesIO() as buffer:
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            buffer.seek(0)
            st.download_button(
                label="Download Excel",
                data=buffer,
                file_name='batch_processing_results.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )
    else:
        st.write("Please upload resumes for batch processing")
