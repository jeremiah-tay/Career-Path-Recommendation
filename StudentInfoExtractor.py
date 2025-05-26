import pandas as pd
import numpy as np
import spacy
import ast
from dateutil import parser
import re
from spacy.matcher import Matcher
from rapidfuzz import fuzz, process
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor

class StudentInfoExtractor:
    def __init__(self, text):
        self.nlp = spacy.load("en_core_web_lg")
        self.skill_extractor = SkillExtractor(self.nlp, SKILL_DB, PhraseMatcher)
        self.soft_skills_list = self.load_soft_skills_list()
        self.text = text
    
    # Rule-based Heuristics: Assumes that the name is in the first line, contains at least 2 words, and it starts with a captial letter
    def extract_name_from_top_line(self, text):
        lines = text.strip().splitlines()
        if not lines:
            return None

        first_line = lines[0].strip()  # remove leading/trailing spaces

        # Check if it's likely to be a name: 2+ words, all start with uppercase
        if len(first_line.split()) >= 2 and all(w[0].isupper() for w in first_line.split() if w[0].isalpha()):
            return first_line.strip()  # just in case, double-strip

        return None
    
    # Regular Expression
    def extract_emails(self, text):
        pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}"
        matches = re.findall(pattern, text)
        return matches if matches else None
    
    # Regular Expressions
    def extract_contact_information(self, text):
        contact_number = None

        # Updated pattern to match international numbers with + sign and spacing
        pattern = r'(?:\+?\d{1,3}[-.\s]?)?(?:\d{3,4}[-.\s]?){2,3}\d{2,4}'

        match = re.search(pattern, text)
        if match:
            contact_number = match.group().strip()
        return contact_number
    
    def capture_education_level(self, text):
        text = text.lower()

        if "phd" in text or "doctor of philosophy" in text:
            return "PhD"
        elif "master" in text or "masters" in text or "master's" in text or "msc" in text:
            return "Master"
        elif "bachelor" in text or "bachelor's" in text or "bsc" in text:
            return "Bachelor"
        elif "polytechnic" in text or "poly" in text:
            return "Polytechnic"
        elif "diploma" in text:
            return "Diploma"
        elif "high school" in text or "junior college" in text or "jc" in text:
            return "High School"
        else:
            return None
        
    def capture_degree_field(self, text):
        pattern = r'(?:Bachelor|Master|PhD|Bachelors|Bachelor\'s|Master\'s|Diploma|Polytechnic|Higher Diploma|Advanced Diploma)[^.\n]*? in ([A-Za-z &\-\/]+)'

        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            degree_field = match.group(1).strip()
            return degree_field

        return None
    
    def capture_university_name(self, text):
        top_290_universities = [
        "Massachusetts Institute of Technology (MIT)",
        "Imperial College London",
        "University of Oxford",
        "Harvard University",
        "University of Cambridge",
        "Stanford University",
        "ETH Zurich – Swiss Federal Institute of Technology",
        "National University of Singapore (NUS)",
        "University College London (UCL)",
        "California Institute of Technology (Caltech)",
        "University of Pennsylvania",
        "University of California, Berkeley (UCB)",
        "The University of Melbourne",
        "Peking University",
        "Nanyang Technological University, Singapore (NTU Singapore)",
        "Cornell University",
        "The University of Hong Kong",
        "The University of Sydney",
        "The University of New South Wales (UNSW Sydney)",
        "Tsinghua University",
        "University of Chicago",
        "Princeton University",
        "Yale University",
        "Université PSL",
        "University of Toronto",
        "EPFL – École polytechnique fédérale de Lausanne",
        "The University of Edinburgh",
        "Technical University of Munich",
        "McGill University",
        "Australian National University (ANU)",
        "Seoul National University",
        "Johns Hopkins University",
        "The University of Tokyo",
        "Columbia University",
        "The University of Manchester",
        "The Chinese University of Hong Kong (CUHK)",
        "Monash University",
        "University of British Columbia",
        "Fudan University",
        "King's College London",
        "The University of Queensland",
        "University of California, Los Angeles (UCLA)",
        "New York University (NYU)",
        "University of Michigan-Ann Arbor",
        "Shanghai Jiao Tong University",
        "Institut Polytechnique de Paris",
        "The Hong Kong University of Science and Technology",
        "Zhejiang University",
        "Delft University of Technology",
        "Kyoto University",
        "Northwestern University",
        "The London School of Economics and Political Science (LSE)",
        "KAIST - Korea Advanced Institute of Science & Technology",
        "University of Bristol",
        "University of Amsterdam",
        "Yonsei University",
        "The Hong Kong Polytechnic University",
        "Carnegie Mellon University",
        "Ludwig-Maximilians-Universität München",
        "Universiti Malaya (UM)",
        "Duke University",
        "City University of Hong Kong",
        "KU Leuven",
        "Sorbonne University",
        "The University of Auckland",
        "University of Texas at Austin",
        "Korea University",
        "National Taiwan University (NTU)",
        "The University of Warwick",
        "University of Illinois at Urbana-Champaign",
        "Universidad de Buenos Aires (UBA)",
        "University of California, San Diego (UCSD)",
        "Université Paris-Saclay",
        "KTH Royal Institute of Technology",
        "Lund University",
        "University of Washington",
        "The University of Western Australia",
        "University of Glasgow",
        "Brown University",
        "University of Birmingham",
        "University of Southampton",
        "The University of Adelaide",
        "University of Leeds",
        "Universität Heidelberg",
        "Tokyo Institute of Technology (Tokyo Tech)",
        "Osaka University",
        "Trinity College Dublin, The University of Dublin",
        "University of Technology Sydney",
        "Durham University",
        "Pennsylvania State University",
        "Purdue University",
        "Universidade de São Paulo",
        "Pontificia Universidad Católica de Chile (UC)",
        "Lomonosov Moscow State University",
        "Universidad Nacional Autónoma de México (UNAM)",
        "University of Alberta",
        "Freie Universitaet Berlin",
        "Pohang University of Science And Technology (POSTECH)",
        "RWTH Aachen University",
        "University of Copenhagen",
        "King Fahd University of Petroleum & Minerals (KFUPM)",
        "Karlsruhe Institute of Technology (KIT)",
        "Uppsala University",
        "University of St Andrews",
        "The University of Sheffield",
        "Utrecht University",
        "Tohoku University",
        "Boston University",
        "University of Nottingham",
        "Technical University of Denmark",
        "University of Zurich",
        "Politecnico di Milano",
        "Aalto University",
        "Georgia Institute of Technology",
        "University of Waterloo",
        "University of Wisconsin-Madison",
        "University of Helsinki",
        "Indian Institute of Technology Bombay (IITB)",
        "University of Oslo",
        "Queen Mary University of London",
        "Western University",
        "Qatar University",
        "RMIT University",
        "Sungkyunkwan University (SKKU)",
        "University of Southern California",
        "Humboldt-Universität zu Berlin",
        "University College Dublin",
        "Stockholm University",
        "Newcastle University",
        "University of California, Davis",
        "University of Basel",
        "Sapienza University of Rome",
        "Alma Mater Studiorum - Università di Bologna",
        "Macquarie University",
        "University of Science and Technology of China",
        "Eindhoven University of Technology",
        "University of Vienna",
        "Universiti Kebangsaan Malaysia (UKM)",
        "Chalmers University of Technology",
        "Universidad de Chile",
        "Lancaster University",
        "Leiden University",
        "Rice University",
        "University of Bern",
        "University of Groningen",
        "University of Pittsburgh",
        "University of Reading",
        "University of Twente",
        "University of York",
        "Vrije Universiteit Amsterdam",
        "Wageningen University & Research",
        "Aarhus University",
        "Arizona State University",
        "Autonomous University of Barcelona",
        "Birkbeck, University of London",
        "Brandeis University",
        "Case Western Reserve University",
        "Chiba University",
        "Colorado State University",
        "Curtin University",
        "Dalhousie University",
        "Deakin University",
        "Drexel University",
        "Ecole des Ponts ParisTech",
        "Ecole Normale Supérieure de Lyon",
        "Ecole Polytechnique",
        "Emory University",
        "Florida State University",
        "George Washington University",
        "Ghent University",
        "Griffith University",
        "Hanyang University",
        "Heriot-Watt University",
        "Hokkaido University",
        "Indian Institute of Science",
        "Indian Institute of Technology Delhi (IITD)",
        "Indian Institute of Technology Kanpur (IITK)",
        "Indian Institute of Technology Kharagpur (IITKGP)",
        "Indian Institute of Technology Madras (IITM)",
        "Indian Institute of Technology Roorkee (IITR)",
        "Indiana University Bloomington",
        "Iowa State University",
        "James Cook University",
        "Jilin University",
        "Kobe University",
        "Kyushu University",
        "La Trobe University",
        "Laval University",
        "Lomonosov Moscow State University",
        "Louisiana State University",
        "Lund University",
        "Mahidol University",
        "Michigan State University",
        "Nagoya University",
        "National Autonomous University of Mexico (UNAM)",
        "National Cheng Kung University",
        "National Chiao Tung University",
        "National Tsing Hua University",
        "Newcastle University",
        "North Carolina State University",
        "Norwegian University of Science and Technology",
        "Ohio State University",
        "Oregon State University",
        "Osaka University",
        "Peking University",
        "Pennsylvania State University",
        "Politecnico di Torino",
        "Pontificia Universidad Católica de Chile",
        "Purdue University",
        "Queen's University",
        "Rensselaer Polytechnic Institute",
        "RMIT University",
        "Rutgers University–New Brunswick",
        "San Diego State University",
        "Seoul National University",
        "Shanghai Jiao Tong University",
        "Sichuan University",
        "Simon Fraser University",
        "Sofia University",
        "Sogang University",
        "South China University of Technology",
        "Southern Methodist University",
        "Stellenbosch University",
        "Stony Brook University",
        "Sungkyunkwan University",
        "Sun Yat-sen University",
        "Syracuse University",
        "Technical University of Munich",
        "Technion - Israel Institute of Technology",
        "Technische Universität Berlin",
        "Technische Universität Dresden",
        "Texas A&M University",
        "The Chinese University of Hong Kong",
        "The Hong Kong Polytechnic University",
        "The Hong Kong University of Science and Technology",
        "The University of Adelaide",
        "The University of Auckland",
        "The University of Hong Kong",
        "The University of Melbourne",
        "The University of New South Wales",
        "The University of Queensland",
        "The University of Sydney",
        "The University of Tokyo",
        "The University of Western Australia",
        "Tianjin University",
        "Tohoku University",
        "Tokyo Institute of Technology",
        "Tongji University",
        "Tsinghua University",
        "Tufts University",
        "Tulane University",
        "UCL (University College London)",
        "Universidad Autónoma de Madrid",
        "Universidad de Buenos Aires",
        "Universidad de Chile",
        "Universidad Nacional Autónoma de México",
        "Universidade de São Paulo",
        "Université de Montréal",
        "Université Laval",
        "Université Paris-Saclay",
        "Université PSL",
        "University College Dublin",
        "University of Alberta",
        "University of Amsterdam",
        "University of Arizona",
        "University of Basel",
        "University of Bath",
        "University of Bergen",
        "University of Birmingham",
        "University of Bristol",
        "University of British Columbia",
        "University of Calgary",
        "University of California, Berkeley",
        "University of California, Davis",
        "University of California, Irvine",
        "University of California, Los Angeles",
        "University of California, San Diego",
        "University of California, Santa Barbara",
        "University of Cambridge",
        "University of Cape Town",
        "University of Colorado Boulder",
        "University of Copenhagen",
        "University of Edinburgh",
        "University of Florida",
        "University of Geneva",
        "University of Glasgow",
        "University of Göttingen",
        "University of Groningen",
        "University of Helsinki",
        "University of Hong Kong",
        ]
        
        lines = text.splitlines()
        best_match = process.extractOne(
            query=' '.join(lines),
            choices = top_290_universities,
            scorer = fuzz.partial_ratio,
            score_cutoff = 80  # Only return if match is strong enough
        )

        if best_match:
            return best_match[0]  # university name
        return None
    
    def capture_gpa_or_classification(self, text):
        # 1. Match GPA (e.g., GPA: 4.5)
        gpa_pattern = r"GPA\s*[:\-]?\s*(\d\.\d{1,2})"
        match = re.search(gpa_pattern, text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}"

        # 2. Match Honours / Merit classifications
        honours_pattern = (
            r"(Honours with Highest Distinction|Honors with Highest Distinction|"
            r"Honours with Distinction|Honors with Distinction|"
            r"Honours with Merit|Honors with Merit|"
            r"Honours|Honors|"
            r"Graduated with Merit|with Merit)"
        )
        match = re.search(honours_pattern, text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}"

        # 3. Match graduation without honours
        no_honours_pattern = r"(without Honours|without Honors|non-honours|non-honors)"
        match = re.search(no_honours_pattern, text, re.IGNORECASE)
        if match:
            return "Graduated without Honours"
        return None
    
    def extract_work_experience_dates(self, text):
        # Step 1: Get work experience section only
        work_section = ""
        lines = text.splitlines()
        in_work_section = False

        for line in lines:
            if "work experience" in line.lower() or "work" in line.lower() or "experience" in line.lower():
                in_work_section = True
                continue
            if in_work_section and (line.strip().lower().startswith("education") or "project" in line.lower()):
                break
            if in_work_section:
                work_section += line + "\n"

        # Step 2: Extract date ranges like "Sep 2024 – Jan 2025"
        date_pattern = re.findall(r'([A-Za-z]+\s\d{4})\s*[-–]\s*([A-Za-z]+\s\d{4})', work_section)

        total_months = 0
        for start_str, end_str in date_pattern:
            try:
                start_date = parser.parse(start_str)
                end_date = parser.parse(end_str)
                months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
                total_months += months
            except:
                continue

        # Convert months to years
        total_years = round(total_months / 12, 1)
        return total_years
    
    def capture_hard_skills(self, text):
        skill_extractor = self.skill_extractor
        annotations = skill_extractor.annotate(text)
        full_matches = annotations['results']['full_matches']
        hard_skill_phrases = set() # to avoid duplicates

        for item in full_matches:
            hard_skill_phrases.add(item['doc_node_value'])

        hard_skill_list = sorted(hard_skill_phrases)

        return hard_skill_list
    
    def load_soft_skills_list(self):
        return [
        "communication", "teamwork", "problem solving", "adaptability", "leadership",
        "creativity", "empathy", "work ethic", "critical thinking", "interpersonal skills",
        "time management", "attention to detail", "collaboration", "resilience", "flexibility",
        "self-motivation", "integrity", "decision-making", "emotional intelligence", "organization",
        "accountability", "conflict resolution", "stress management", "public speaking", "negotiation",
        "patience", "active listening", "persuasion", "cultural awareness", "constructive feedback",
        "initiative", "self-awareness", "positivity", "goal setting", "multitasking",
        "mentoring", "influence", "diplomacy", "respectfulness", "helpfulness",
        "accepting feedback", "tolerance", "relationship building", "customer focus", "self-regulation",
        "team building", "managing ambiguity", "perspective-taking", "coaching", "follow-through",
        "professionalism", "humility", "dependability", "result orientation", "receptiveness",
        "curiosity", "collaborative mindset", "resourcefulness", "enthusiasm", "dedication",
        "learning agility", "perseverance", "grit", "open-mindedness", "confidence",
        "clarity", "fairness", "courtesy", "approachability", "perspicacity",
        "mindfulness", "tact", "observational skills", "analytical thinking", "prioritization",
        "risk-taking", "strategic thinking", "proactiveness", "goal orientation", "concentration",
        "body language awareness", "presentation skills", "inclusiveness", "team spirit", "loyalty",
        "self-confidence", "service orientation", "engagement", "civic-mindedness", "discipline",
        "cooperation", "innovation", "pragmatism", "emotional regulation", "humor",
        "inspirational skills", "ethical judgment", "self-reflection", "being a good listener", "decisiveness",
        "cross-functional collaboration", "reliability", "respect for diversity", "multicultural competence", "confidentiality",
        "time awareness", "efficiency", "sociability", "brainstorming", "inquiry",
        "context awareness", "diligence", "goal alignment", "people management", "service mindset",
        "networking", "stress tolerance", "discipline", "optimism", "tenacity",
        "giving recognition", "peer support", "peer learning", "team synergy", "intrapersonal skills",
        "systems thinking", "respect for others’ time", "customer-centricity", "self-discipline", "project ownership",
        "volunteering spirit", "internal motivation", "problem sensitivity", "personal initiative", "conflict management",
        "clarity in writing", "initiative-taking", "constructive criticism", "mental flexibility", "learning from failure",
        "engaging others", "facilitation skills", "resolving ambiguity", "diversity sensitivity", "negotiation skills",
        "peer mentoring", "boundary-setting", "social awareness", "building rapport", "relationship management",
        "peer collaboration", "cohesiveness", "respectful disagreement", "team facilitation", "active contribution",
        "learning from feedback", "conversational skills", "resilience under pressure", "persistence", "evaluative thinking",
        "analytical listening", "recognizing bias", "maintaining composure", "ethical thinking", "managing disappointment",
        "seeking help", "managing expectations", "setting boundaries", "vision articulation", "persuasive communication",
        "providing encouragement", "giving constructive criticism", "receiving criticism well", "avoiding gossip", "trustworthiness",
        "attention management", "scheduling", "habit formation", "workflow optimization", "time blocking",
        "crisis management", "adjusting priorities", "self-reinforcement", "self-appraisal", "welcoming change",
        "mental clarity", "social perceptiveness", "awareness of nonverbal cues", "emotional resilience", "adaptive learning",
        "clarifying assumptions", "welcoming feedback", "growth mindset", "sensemaking", "giving appreciation",
        "bridging communication gaps", "authenticity", "initiative in learning", "consensus building", "delegation",
        "mobilizing teams", "boundary management", "value alignment", "feedback looping", "improvisation",
        "time estimation", "respect for protocols", "collaborative problem solving", "prioritizing well-being", "nurturing others",
        "goal visualization", "trust building", "role flexibility", "proximity management", "credibility",
        "task ownership", "developing others", "preventing burnout", "leading by example", "peer evaluation",
        "handling rejection", "contextual thinking", "personal branding", "emotional balance", "listening with empathy",
        "assertiveness", "goal tracking", "conflict prevention", "solution orientation", "boundary awareness",
        "change management", "attention to social cues", "understanding group dynamics", "decision confidence", "value-driven behavior",
        "habitual consistency", "peer encouragement", "reframing", "handling failure", "modeling integrity",
        "noticing others’ strengths", "leveraging diversity", "self-compassion", "critiquing ideas not people", "respect for hierarchy",
        "adaptation to feedback", "reducing misunderstandings", "thinking before speaking", "soft assertiveness", "non-defensive responses",
        "maintaining focus", "quiet leadership", "noticing patterns", "timing your input", "emotional containment",
        "asking meaningful questions", "strategic compromise", "positive reinforcement", "speaking up", "discretion",
        "peer empowerment", "energizing others", "knowing your audience", "connecting ideas", "avoiding distractions",
        "team accountability", "handling ethical dilemmas", "co-creation", "learning how to learn", "transparent communication"
        ]
    
    def capture_soft_skills(self, text, similarity_threshold = 0.7):
        nlp = self.nlp  # Make sure this model is installed!
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(skill) for skill in self.soft_skills_list]
        matcher.add("SOFT_SKILLS", patterns)

        doc = nlp(text)

        # --- Phrase Matching ---
        matched_skills = set()
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            matched_skills.add(span.text.lower())

        # --- Vector Similarity Matching ---
        for skill in self.soft_skills_list:
            skill_doc = nlp(skill)
            # Check if both documents have vectors
            if doc.has_vector and skill_doc.has_vector:
                similarity = doc.similarity(skill_doc)
                if similarity >= similarity_threshold:
                    matched_skills.add(skill.lower())

        return sorted(matched_skills)
    
    def extract_all_info(self):
        try:
            return pd.DataFrame([{
                'Name': self.extract_name_from_top_line(self.text) or '',
                'Email': self.extract_emails(self.text) or [],
                'Contact Information': self.extract_contact_information(self.text) or '',
                'Education Level': self.capture_education_level(self.text) or '',
                'Degree Field': self.capture_degree_field(self.text) or '',
                'University': self.capture_university_name(self.text) or '',
                'GPA': self.capture_gpa_or_classification(self.text) or '',
                'Work Experience': self.extract_work_experience_dates(self.text) or 0,
                'Hard Skills': self.capture_hard_skills(self.text) or [],
                'Soft Skills': self.capture_soft_skills(self.text) or []
            }])
        except Exception as e:
            print(f"[ERROR] Failed to extract info: {e}")
            return pd.DataFrame()