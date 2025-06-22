# 🎓 Career-Path-Recommendation
An AI-powered recommendation engine that analyzes university students’ resumes and academic backgrounds to suggest tailored career paths based on their skills, education, and experience using Natural Language Processing (NLP) and semantic matching.

## 📌 Overview
This project aims to help students make informed career choices by analyzing their resumes and recommending job roles that align with their:
- Hard and soft skills
- Educational background
- Work experience
The system leverages Natural Language Processing (NLP), semantic embeddings, and machine learning to extract, structure, and compare student profiles against a curated dataset of job opportunities.

## 💡 Features
- 🧠 **Resume Parsing**: Extract structured information (skills, education, experience) from PDF resumes using NLP techniques.
- ✨ **Semantic Skill Matching**: Uses sentence embeddings to compute similarity between student profiles and job descriptions.
- 🤖 **Clustering-Based Recommendation**: Optionally cluster similar jobs and match students within the most relevant group.
- 🔍 **Top-N Job Recommendations**: Recommend the top 5 most relevant jobs for each student.
- 📊 **PostgreSQL Integration**: Automatically stores parsed profiles and job recommendations in a PostgreSQL database.

## 📁 Project Structure
```
career-path-recommendation/
├── resume/                                      # Uploaded student resumes
├── job_data.csv                                 # Curated job role dataset
├── RecommendationProcessor.py                   # Semantic scoring recommendation algorithm
├── ClusterProcessor.py                          # Clustering-based recommendation algorithm
├── StudentInfoExtractor.py                      # Resume extraction and parsing
├── databaseProcessor.py                         # PostgreSQL data insertion
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
    ```python
   if __name__ == "__main__":
    resume_pdf_file = "John Doe Resume.pdf"
    text = extract_text(f"./resume/{resume_pdf_file}")
    main(text, top_k = 5, algorithm = "Clustering")
3. Run the pipeline on your terminal:
   ```
   python main.py
   ```
4. The pipeline will:
   - Parse the resume using NLP
   - Extract structured data (Education, Experience, Skills)
   - Compute similarity scores between the students and the available jobs)
   - Recommend the top 5 jobs
   - Insert both the student profile and recommendations into your PostgreSQL database

### 🤖 Algorithms
You can switch the recommendation engine using the `algorithm` parameter at the bottom of the `main.py` file.

| Algorithm Name | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `Semantic`     | Calculates a weighted semantic similarity score between student and jobs.   |
| `Clustering`   | Uses KMeans to group similar jobs and recommend top matches in the cluster. |

## 📦 Sample Output
```java
Top 5 Recommended Jobs for John Doe:

Machine Learning Engineer: 0.701
Data Scientist: 0.701
AI Product Manager: 0.683
Prompt Engineer (for AI): 0.664
Mobile App Developer: 0.663




