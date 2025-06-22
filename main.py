import argparse
import pandas as pd
from pdfminer.high_level import extract_text
from StudentInfoExtractor import StudentInfoExtractor
from RecommendationProcessor import RecommendationProcessor
from databaseProcessor import databaseProcessor
from ClusterProcessor import ClusterProcessor

def main(text, top_k = 5, algorithm = "semantic"):
    # Load job dataset (hardcoded path)
    job_df = pd.read_csv("job_data.csv")

    # Extract structured student info
    extractor = StudentInfoExtractor(text)
    nlp_df = extractor.extract_all_info()
    print("✅ Student info extracted")
    print(nlp_df.iloc[0])

    if nlp_df.empty:
        print("❌ No student information could be extracted from the resume.")
        return
    
    # Insert student's recommendation into the database
    db_processor = databaseProcessor(nlp_df)
    db_processor.insert_student()
    print("✅ Student inserted into database")
    
    # Use the most recent student profile
    student_profile = nlp_df.iloc[-1]

    # Recommend top 5 jobs
    if algorithm == "Semantic":
        print("Recommending Jobs using semantic matching...")
        recommender = RecommendationProcessor(student_profile, job_df)
        recommended_jobs = recommender.recommend_top_jobs(top_k)
        new_student = recommender.generate_recommendation_row(recommender.student.Name, recommended_jobs)

    
    elif algorithm == "Clustering":
        print("Recommending Jobs using clustering matching...")
        processor = ClusterProcessor(student_profile, job_df)
        processor.compute_job_score()
        recommended_jobs = processor.recommend_top_k(k = top_k)
        new_student = processor.generate_recommendation_row(processor.student.Name, recommended_jobs)

    # Insert student's recommendaiton into the database
    db_processor.insert_job(new_student)
    return recommended_jobs

if __name__ == "__main__":
    resume_pdf_file = "Tay Zhi Wen Jeremiah CV.pdf"
    text = extract_text(f"./resume/{resume_pdf_file}")
    main(text, top_k = 5, algorithm = "Clustering")
