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
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input, pdf_content, prompt])
    return response.text

@st.cache_data()
def get_gemini_response_keywords(input, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input, pdf_content, prompt])
    return response.text

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
    skills = {
        "Technical Skills": keywords.get('Technical Skills', []),
        "Analytical Skills": keywords.get('Analytical Skills', []),
        "Soft Skills": keywords.get('Soft Skills', []),
    }
    
    fig = px.bar(skills, title="Skills Gap Visualization")
    st.plotly_chart(fig)

def calculate_match_score(response_text):
    # Extract match score from the response (assuming response includes a match score percentage)
    try:
        match_score = float(response_text.split("Match Score:")[1].strip().split("%")[0])
        return match_score
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
        if file.type == "application/pdf":
            resume_content = input_pdf_setup(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_content = input_docx_setup(file)
        response = analyze_resume(resume_content, job_description)
        match_score = calculate_match_score(response)
        results.append({'File Name': file.name, 'Match Score': match_score, 'Response': response})
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
Extract keywords from the resume and categorize them into:
1. Technical Skills
2. Analytical Skills
3. Soft Skills
"""

input_prompt3 = """
You are an advanced ATS. Provide a percentage match score between the resume content and job description considering:
1. Skills Weight System: Differentiate between must-have and nice-to-have skills.
2. Experience Level Matching: Consider years of experience in skill matching.
3. Education Requirements Matching: Compare candidate qualifications with job requirements.
4. Project Experience Analysis: Match project experiences with job requirements.
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
            st.write(f"Match Score: {result['Match Score']}%")
            st.write(result['Response'])
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
