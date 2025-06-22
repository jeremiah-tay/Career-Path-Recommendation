import streamlit as st
import numpy as np
import pandas as pd
import os
import json
import time
import argparse
import base64
from docx2pdf import convert
from pdfminer.high_level import extract_text
from StudentInfoExtractor import StudentInfoExtractor
from RecommendationProcessor import RecommendationProcessor
from databaseProcessor import databaseProcessor
from ClusterProcessor import ClusterProcessor
from main import main

# Title
st.set_page_config(page_title = "AI-Career Path Recommender", page_icon = "📚", layout = "wide")
st.title("🎓 AI-Powered Career Path Recommendation System")

# Sidebar
st.sidebar.header("Configurations")
algorithm = st.sidebar.selectbox(
    "Choose Preferred Recommendation Algorithm",
    ("Semantic", "Clustering")
    )
top_k = st.sidebar.slider(
    "Top k Job Recommendations",
    min_value = 1, max_value = 10, value = 5
)

# Uploading File
uploaded_file = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])
if uploaded_file is not None:
    if not uploaded_file.name.lower().endswith(".pdf"):
        st.error("❌ Invalid file format. Please upload a PDF file.")
    resume_path = uploaded_file.name
    st.success("✅ PDF resume uploaded successfully!")

    # Extract text directly from in-memory file object
    text = extract_text(uploaded_file)
    show_resume = st.checkbox("My Resume")
    if show_resume:
        st.markdown("### 📄 Extracted Text")
        st.text_area(
        label = "📄 Extracted Text",
        value = text,
        label_visibility = "collapsed",
        height = 300
        )
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col3:
        if st.button("🚀 Start Recommending Process"):
            with st.spinner("🔍 Recommending jobs..."):
                recommended_jobs = main(text, top_k = top_k, algorithm = algorithm)
                time.sleep(1)  # Optional: simulate loading delay

            st.success("✅ Job Recommendation Successful!")
            
            if recommended_jobs is not None:
                if algorithm == "Clustering":
                    st.subheader("📌 Recommended Jobs")
                    st.dataframe(recommended_jobs[["Job Title", "similarity"]].rename(columns={"similarity": "Score"}), hide_index=True)
                elif algorithm == "Semantic":
                    st.subheader("📌 Recommended Jobs")
                    st.dataframe(recommended_jobs[["Job Title", "Total Score"]].rename(columns={"Total Score": "Score"}), hide_index=True)
           