# 🎓 Career-Path-Recommendation
An AI-powered recommendation engine that analyzes university students’ resumes and academic backgrounds to suggest tailored career paths based on their skills, education, and experience using Natural Language Processing (NLP) and semantic matching.

## 📌 Overview
This project aims to help students make informed career choices by analyzing their resumes and recommending job roles that align with their:
- Hard and soft skills
- Educational background
- Work experience
It uses Natural Language Processing (NLP), semantic embeddings, and machine learning to extract, structure, and compare student profiles against a curated dataset of job opportunities.

## 💡 Features
- 🧠 **Resume Parsing**: Extract structured information (skills, education, experience) from PDF resumes using NLP.
- ✨ **Semantic Skill Matching**: Use sentence embeddings to compute similarity between student profiles and job requirements.
- 🔍 **Top-N Job Recommendations**: Recommend the top 5 most relevant jobs for each student.
- 📊 **PostgreSQL Integration**: Store parsed profiles and recommendation results in a relational database.

## 📁 Project Structure
```
career-path-recommendation/
├── resume/                     # Uploaded student resumes
├── job_data.csv               # Curated job role dataset
├── recommendation.py          # Main recommendation engine
├── extractor.py               # Resume parsing logic
├── database.py                # PostgreSQL data insertion
├── app.py / main.py           # Entry point
├── README.md                  # Project documentation
```
