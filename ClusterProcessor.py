import ast
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class ClusterProcessor:
    def __init__(self, student_profile, job_data, n_clusters: int = 5):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.student = student_profile
        self.job_data = job_data.copy()
        self.kmeans = KMeans(n_clusters = n_clusters, random_state = 42)

        # Preprocess skills
        self.job_data["Hard Skills"] = self.job_data["Hard Skills"].apply(ast.literal_eval)
        self.job_data["Soft Skills"] = self.job_data["Soft Skills"].apply(ast.literal_eval)
        self.job_data['Hard Skills'] = self.job_data['Hard Skills'].apply(self.preprocess_skills)
        self.job_data['Soft Skills'] = self.job_data['Soft Skills'].apply(self.preprocess_skills)
    
    def preprocess_skills(self, skills):
        return [skill.strip().lower() for skill in skills]
    
    def embed_job(self, job_row):
        """
        Embeds a job posting into a vector using SentenceTransformer.
        """
        text = (
            f"Job Title: {job_row['Job Title']}. "
            f"Industry: {job_row['Industry']}. "
            f"Hard Skills required: {', '.join(job_row['Hard Skills'])}. "
            f"Soft Skills required: {', '.join(job_row['Soft Skills'])}. "
            f"Required degree field: {job_row['Required Degree Field']}. "
            f"Required education level: {job_row['Required Education']}. "
            f"Years of experience required: {job_row['Years of Experience']}."
        )
        return self.model.encode(text, convert_to_numpy = True)

    
    def embed_student(self):
        """
        Embeds the student profile into a vector using SentenceTransformer.
        """
        text = (
            f"Student profile. "
            f"Hard Skills: {', '.join(self.student['Hard Skills'])}. "
            f"Soft Skills: {', '.join(self.student['Soft Skills'])}. "
            f"Degree field: {self.student['Degree Field']}. "
            f"Education level: {self.student['Education Level']}. "
            f"Work experience: {self.student['Work Experience']} years."
        )
        return self.model.encode(text, convert_to_numpy = True)
    
    def compute_job_score(self):
        """
        Embeds all jobs, clusters them, and returns the jobs in the same cluster
        as the student with a similarity score.
        """
        # Embed jobs
        self.job_data['vectors'] = self.job_data.apply(lambda row: self.embed_job(row), axis = 1)
        job_vectors = np.stack(self.job_data['vectors'].values).astype(np.float64)

        # Embed student
        self.student_vector = self.embed_student().astype(np.float64)
        
        # Fit KMeans
        self.job_data['cluster'] = self.kmeans.fit_predict(job_vectors)

        # Predict student's cluster
        student_cluster = self.kmeans.predict([self.student_vector])[0]

        # Filter jobs in the same cluster
        cluster_jobs = self.job_data[self.job_data['cluster'] == student_cluster].copy()

         # Compute cosine similarities
        cluster_matrix = np.stack(cluster_jobs['vectors'].values)
        similarities = cosine_similarity([self.student_vector], cluster_matrix)[0]
        cluster_jobs['similarity'] = similarities
        
        self.cluster_jobs = cluster_jobs
        return cluster_jobs
            
    def recommend_top_k(self, k = 5):
        """
        Prints the top k most similar jobs to the student.
        """
        if not hasattr(self, 'cluster_jobs'):
            self.compute_job_score()

        top_k_jobs = self.cluster_jobs.sort_values(by = 'similarity', ascending = False).head(k)

        # Display top 5 jobs
        print(f"\nTop {k} Recommended Jobs for {self.student['Name']}:\n\nTotal Score:")
        for _, row in top_k_jobs.iterrows():
            print(f"{row['Job Title']}: {row['similarity']:.3f}")
        
        return top_k_jobs
    
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