import argparse
import pandas as pd
from pdfminer.high_level import extract_text
from StudentInfoExtractor import StudentInfoExtractor
from RecommendationProcessor import RecommendationProcessor
from databaseProcessor import databaseProcessor

def main(resume_pdf_path, top_n = 5):
    # Load job dataset (hardcoded path)
    job_df = pd.read_csv("job_data.csv")

    # Extract resume text
    text = extract_text(f"./resumes/{resume_pdf_path}")

    # Extract structured student info
    extractor = StudentInfoExtractor(text)
    nlp_df = extractor.extract_all_info()
    print(nlp_df.iloc[0])

    if nlp_df.empty:
        print("❌ No student information could be extracted from the resume.")
        return
    
    # Insert student information into the database
    db_processor = databaseProcessor(nlp_df)
    db_processor.insert_student()
    
    # Use the most recent student profile
    student_profile = nlp_df.iloc[-1]

    # Recommend top 5 jobs
    print("Recommending Jobs...")
    recommender = RecommendationProcessor(student_profile, job_df)
    recommended_jobs = recommender.recommend_top_jobs(top_n)
    new_student = recommender.generate_recommendation_row(recommender.student.Name, recommended_jobs)

    # Insert student's recommendaiton into the database
    db_processor.insert_job(new_student)

if __name__ == "__main__":
    main("Tay Zhi Sheng Japheth John - Resume.pdf", top_n = 5)
