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
├── resume/                                      # Uploaded student resumes
├── job_data.csv                                 # Curated job role dataset
├── RecommendationProcessor.py                   # Recommendation engine
├── StudentInfoExtractor.py                      # Resume extraction and parsing
├── database.py                                  # PostgreSQL data insertion
├── main.py                                      # Main pipeline file
├── recommender_system_working_notebook.ipynb    # Project Rough Working
├── README.md                                    # Project documentation
```
## 🚀 Getting Started
### ✅ Prerequisites
- Python 3.8+
- PostgreSQL running with credentials configured
- Create a .env file with the following keys:
```
DB_HOST=
DB_NAME=
DB_USER=
DB_PASSWORD=
DB_PORT=
```
### 🔧 Installation
```
git clone https://github.com/jeremiah-tay/Career-Path-Recommendation.git
cd Career-Path-Recommendation
```
## 🧪 How It Works
1. Place your resume PDF into the ```resume/``` folder (e.g., ```resume/John Doe Resume.pdf```)
2. Open the main.py script and scroll to the bottom. Edit the following lines to point to your resume file:
   ```
   if __name__ == "__main__":
       main("resume/John Doe Resume.pdf", top_n=5)
  ```
3. Run the main script:
   ```
   python main.py
   ```
4. The pipeline will:
   - Parse the resume using NLP
   - Extract structured data (Education, Experience, Skills)
   - Compute similarity scores between the students and the available jobs)
   - Insert both the student profile and recommendations into your PostgreSQL database




