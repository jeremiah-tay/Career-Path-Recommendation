import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from psycopg2.extras import Json

class databaseProcessor:
    def __init__(self, df):
        self.df = df
    
    def insert_student(self):
        # Load environment variables
        load_dotenv(override=True)

        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "5432")
        )
        # Create cursor
        cur = conn.cursor()

        query = '''
            INSERT INTO student (
                name,
                email,
                contact_information,
                education_level,
                degree_field,
                university,
                gpa,
                work_experience_years,
                hard_skills,
                soft_skills
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
        '''

        # Insert row-by-row
        for _, row in self.df.iterrows():
            try:
                cur.execute(query, (
                    row.get("Name"),
                    row.get("Email")[0] if isinstance(row.get("Email"), list) else row.get("Email"),
                    row.get("Contact Information"),
                    row.get("Education Level"),
                    row.get("Degree Field"),
                    row.get("University"),
                    row.get("GPA"),
                    float(row["Work Experience"]) if pd.notna(row["Work Experience"]) else None,
                    Json(row.get("Hard Skills", [])),
                    Json(row.get("Soft Skills", []))
                ))
            except Exception as e:
                print(f"Error inserting row: {row.to_dict()}")
                print("Exception:", e)

        # Commit and close
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Insert student complete.")
    
    def insert_job(self, student):
        # Load environment variables
        load_dotenv(override = True)

        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "5432")
        )
        # Create cursor
        cur = conn.cursor()

        query = '''
            INSERT INTO recommendation (
                name,
                first_recommendation,
                second_recommendation,
                third_recommendation,
                fourth_recommendation,
                fifth_recommendation
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
        '''


        cur.execute(query, (
            student["name"],
            student["first_recommendation"],
            student["second_recommendation"],
            student["third_recommendation"],
            student["fourth_recommendation"],
            student["fifth_recommendation"]
        ))
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Insert job recommendation complete.")