# 🚀 AI-Powered Applicant Tracking System (ATS)

## 📝 Project Overview

This AI-driven Applicant Tracking System (ATS) is a cutting-edge solution designed to revolutionize the recruitment process by providing intelligent resume analysis and matching capabilities.

![Cold Email Generator Interface](https://i.ibb.co/18cw3PG/Screenshot-2024-12-16-204621.png)

## ✨ Key Features

### 1. Intelligent Resume Analysis
- Upload and analyze resumes in PDF and DOCX formats
- Extract and categorize skills
- Provide comprehensive resume feedback

### 2. Job Description Matching
- Compare resume content against job descriptions
- Generate detailed match scores
- Visualize skill compatibility

### 3. Multi-Resume Batch Processing
- Upload and analyze multiple resumes simultaneously
- Export results in CSV and Excel formats
- Instant skill and match score comparisons

## 🛠 Technologies Used

- **AI**: Google Gemini 1.5 Flash
- **Web Framework**: Streamlit
- **Data Processing**: 
  - Pandas
  - PyMuPDF
  - python-docx
- **Visualization**: 
  - Plotly
  - Plotly Express

## 🔍 Detailed Functionality

### Resume Skills Extraction
- Categorizes skills into:
  - Technical Skills
  - Analytical Skills
  - Soft Skills

### Match Score Calculation
- Advanced scoring mechanism considering:
  - Skill weight system
  - Experience level matching
  - Education requirements
  - Project experience analysis

### Visualization
- Interactive skill gap charts
- Percentage match gauge
- Detailed skill breakdown

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Google Gemini API Key

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/ATS-System.git
cd ATS-System
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up Google Gemini API Key
- Create a `.streamlit/secrets.toml` file
- Add your API key:
```toml
GOOGLE_API_KEY = "your_api_key_here"
```

4. Run the application
```bash
streamlit run app.py
```

## 📋 Usage Instructions

1. Enter Job Description
2. Upload Resume(s)
3. Choose Analysis Type:
   - Skills Extraction
   - Percentage Match
   - Batch Processing

## 🔒 Security Notes
- API keys managed via Streamlit secrets
- Secure file handling
- No local storage of sensitive information

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License
[Specify your license here, e.g., MIT License]

## 🌟 Future Roadmap
- Enhanced AI matching algorithms
- More detailed skill taxonomy
- Support for more file formats
- Advanced reporting features

## 💡 Developed By
[Your Name/Organization]

---

**Note**: Ensure you have a valid Google Gemini API key to use this application.
