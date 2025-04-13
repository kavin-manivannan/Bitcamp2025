import streamlit as st
import fitz
from rag import answer

def extract_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype='pdf') as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

st.markdown("""
    <h1 style='text-align: center;'>AI Resume Coach 📝</h1>
    <p style='text-align: center; font-size: 18px;'>Upload your resume. Get instant, AI-powered feedback!</p>
    <hr>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload Your Resume (PDF only)", type=['pdf'])

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    resume_text = extract_pdf(uploaded_file)

    with st.expander("🔍 Preview Resume Text (first 1500 characters)"):
        st.code(resume_text[:1500], language="markdown")

    if st.button("🚀 Generate Feedback"):
        with st.spinner("Generating feedback with AI..."):
            feedback = answer(resume_text)
            with st.expander("📬 Here's Your AI-Powered Feedback"):
                st.write(feedback)
else:
    st.info("Please upload a PDF resume to get started.")