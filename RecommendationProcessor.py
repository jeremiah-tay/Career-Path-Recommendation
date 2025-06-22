import pandas as pd
import numpy as np
import math
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


class RecommendationProcessor:
    def __init__(self, student_profile, job_data):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.student = student_profile
        self.job_data = job_data

        # Preprocess skills
        self.job_data["Hard Skills"] = self.job_data["Hard Skills"].apply(ast.literal_eval)
        self.job_data["Soft Skills"] = self.job_data["Soft Skills"].apply(ast.literal_eval)
        self.job_data['Hard Skills'] = self.job_data['Hard Skills'].apply(self.preprocess_skills)
        self.job_data['Soft Skills'] = self.job_data['Soft Skills'].apply(self.preprocess_skills)
    
    def preprocess_skills(self, skills):
        return [skill.lower().strip() for skill in skills if isinstance(skill, str) and skill.strip()]
    
    def embed_skills(self, skills):
        if not skills or not any(skills):
            return np.zeros(self.model.get_sentence_embedding_dimension())

        embeddings = self.model.encode(skills, convert_to_numpy = True)
        return np.mean(embeddings, axis = 0)
    
    # Cosine similarity between two skill lists
    def skill_cosine_similarity(self, job_skills, student_skills):
        if not job_skills or not student_skills:
            return 0.0  # One empty - no similarity
        
        job_embed = self.embed_skills(job_skills)
        student_embed = self.embed_skills(student_skills)
        cosine_sim = cosine_similarity([job_embed], [student_embed])[0][0]
        return cosine_sim
    
    # Cosine Similarity between Student's Degree Field vs Required Degree Field
    def degree_field_semantic_similarity(self, job_field, student_field):
        embeddings = self.model.encode([job_field, student_field])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def map_education_level(self, level):
        mapping = {
            "High School": 1,
            "Diploma": 2,
            "Polytechnic": 2,
            "Bachelor": 3,
            "Master": 4,
            "PhD": 5
        }
        return mapping.get(level, 0)

    def experience_score(self, student_exp, job_exp):
        diff = student_exp - job_exp
        return 1 / (1 + math.exp(-0.5 * diff))

    def compute_job_score(self, job):
        student = self.student

        education_score = 1 if self.map_education_level(student['Education Level']) >= self.map_education_level(job['Required Education']) else 0
        degree_score = self.degree_field_semantic_similarity(job['Required Degree Field'], student['Degree Field'])
        exp_score = self.experience_score(student['Work Experience'], job['Years of Experience'])
        hard_skill_score = self.skill_cosine_similarity(job['Hard Skills'], student['Hard Skills'])
        soft_skill_score = self.skill_cosine_similarity(job['Soft Skills'], student['Soft Skills'])

        total_score = (
            0.15 * education_score +
            0.20 * degree_score +
            0.20 * exp_score +
            0.30 * hard_skill_score +
            0.15 * soft_skill_score
        )
        result = [
            {
            "Job Title": job["Job Title"],
            "Total Score": round(total_score, 3),
            "Education Score": education_score,
            "Degree Score": round(degree_score, 3),
            "Experience Score": round(exp_score, 3),
            "Hard Skill Score": round(hard_skill_score, 3),
            "Soft Skill Score": round(soft_skill_score, 3)
        }]

        return pd.DataFrame(result)
    
    def recommend_top_jobs(self, top_n = 5):
        df = pd.DataFrame(columns=[
        "Job Title",
        "Total Score",
        "Education Score",
        "Degree Score",
        "Experience Score",
        "Hard Skill Score",
        "Soft Skill Score"
        ])
        for _, job in self.job_data.iterrows():
            score = self.compute_job_score(job)
            df = pd.concat([df, score], ignore_index=True)

        sorted_df = df.sort_values(by="Total Score", ascending=False)

        # Display top 5 jobs
        print(f"\nTop {top_n} Recommended Jobs for {self.student['Name']}:\n")
        for _, row in sorted_df.head(top_n).iterrows():
            print(f"{row['Job Title']}: {row['Total Score']:.3f}")
        return sorted_df.head(top_n)
    
    def generate_recommendation_row(self, student_name, top_jobs):
        titles = top_jobs['Job Title'].tolist()
        return {
            "name": student_name,
            "first_recommendation": titles[0] if len(titles) > 0 else None,
            "second_recommendation": titles[1] if len(titles) > 1 else None,
            "third_recommendation": titles[2] if len(titles) > 2 else None,
            "fourth_recommendation": titles[3] if len(titles) > 3 else None,
            "fifth_recommendation": titles[4] if len(titles) > 4 else None
        }