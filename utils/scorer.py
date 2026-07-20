from sentence_transformers import SentenceTransformer
import torch
from utils.constants import Constants
from datatypes.resume_data_type import Resume
from datatypes.jd_data_type import JobDescription
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import date

class Scorer:
    __model: SentenceTransformer
    __model_name: str

    # Tweakable parameters..
    __certifications_penalty = Constants.CERTIFICATIONS_PENALTY
    __projects_pentalty = Constants.PROJECTS_PENTALTY
    __skill_match_threshold = Constants.SKILL_MATCH_THRESHOLD
    __education_match_threshold = Constants.EDUCATION_MATCH_THRESHOLD
    __certifications_match_threshold = Constants.CERTIFICATIONS_MATCH_THRESHOLD

    def __init__(self, model_name: str, device: str | None = None) -> None:
        print("Scorer :: Inititalizing the model")
        self.__model_name = model_name
        self.__device = device if device else self.__detect_device()
        self.__model = SentenceTransformer(model_name, device=self.__device)
        print(f"Scorer :: Model {self.__model_name} is initiated on device '{self.__device}'")

    @staticmethod
    def __detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


    # summary_match_score: cosine similarity of resume description embedding vs JD job_summary
    # embedding, scaled by the assigned SUMMARY_MATCH_SCORE.
    def get_summary_score(self, resume: Resume, job_description: JobDescription):
        resume_summary = resume.description
        jd_summary = job_description.job_summary

        if not jd_summary:
            return Constants.SUMMARY_MATCH_SCORE
        
        if not resume_summary:
            return 0

        resume_embedding = np.asarray(self.__model.encode(resume_summary))
        jd_embedding = np.asarray(self.__model.encode(jd_summary))

        return cosine_similarity(np.atleast_2d(resume_embedding), np.atleast_2d(jd_embedding))[0][0] * Constants.SUMMARY_MATCH_SCORE
    

    # role_match_score: match each JD role_description embedding against resume experience/
    # project/certification description embeddings, take the max similarity per role
    # (projects and certifications are down-weighted by their penalty factors), then
    # score = (sum of max similarities / total role_descriptions) * assigned ROLE_MATCH_SCORE.
    def get_role_score(self, resume: Resume, job_description: JobDescription):
        jd_roles = job_description.role_description
        resume_descriptions = {
            "experiences": [ " ".join(experience.descriptions) for experience in resume.experience ],
            "projects": [ project.description for project in resume.projects ],
            "certifications": [ cert.description for cert in resume.certifications ]
        }

        if not jd_roles or not len(jd_roles):
            return Constants.ROLE_MATCH_SCORE
        
        if not len(resume_descriptions["experiences"]) and not len(resume_descriptions["projects"]) and not len(resume_descriptions["certifications"]):
            return 0
        
        resume_embeddings = {
            "experiences": np.asarray(self.__model.encode(resume_descriptions["experiences"])),
            "projects": np.asarray(self.__model.encode(resume_descriptions["projects"])),
            "certifications": np.asarray(self.__model.encode(resume_descriptions["certifications"]))
        }

        jd_embeddings = np.asarray(self.__model.encode(jd_roles))

        max_scores = []

        for jd_embedding in jd_embeddings:
            max_score = 0
            if len(resume_embeddings["experiences"]):
                experience_similarities = cosine_similarity(np.atleast_2d(jd_embedding), np.atleast_2d(resume_embeddings["experiences"]))
                max_score = max_score if max_score >= np.max(experience_similarities[0]) else np.max(experience_similarities[0])
            
            if len(resume_embeddings["projects"]):
                project_similarities = cosine_similarity(np.atleast_2d(jd_embedding), np.atleast_2d(resume_embeddings["projects"]))
                project_similarities = project_similarities * self.__projects_pentalty
                max_score = max_score if max_score >= np.max(project_similarities[0]) else np.max(project_similarities[0])

            if len(resume_embeddings["certifications"]):
                cert_similarities = cosine_similarity(np.atleast_2d(jd_embedding), np.atleast_2d(resume_embeddings["certifications"]))
                cert_similarities = cert_similarities * self.__certifications_penalty
                max_score = max_score if max_score >= np.max(cert_similarities[0]) else np.max(cert_similarities[0])

            max_scores.append(max_score)

        return (np.sum(max_scores) / len(jd_roles)) * Constants.ROLE_MATCH_SCORE
    

    # skills_match_score: match JD skill embeddings (required/general/soft, selected by `type`)
    # against resume skills embeddings, count skills whose max similarity is >= skill_match_threshold,
    # score = (matched skill count / total JD skills of this type) * assigned score for the type.
    def get_skills_score(self, resume: Resume, job_description: JobDescription, type: str = "required"):
        resume_skills = resume.skills
        jd_skills = []
        score = 0

        match type:
            case "required": 
                jd_skills = job_description.required_skills
                score = Constants.REQUIRED_SKILL_MATCH_SCORE
            case "general":
                jd_skills = job_description.general_skills
                score = Constants.GENERAL_SKILL_MATCH_SCORE
            case "soft":
                jd_skills = job_description.soft_skills
                score = Constants.SOFT_SKILL_MATCH_SCORE
            case _:
                raise Exception("skills_score :: Invalid type passed")
            
        if not len(jd_skills):
            return score
        
        if not len(resume_skills):
            return 0
            
        jd_embeddings = np.asarray(self.__model.encode(jd_skills))
        resume_embeddings = np.asarray(self.__model.encode(resume_skills))

        similarities = cosine_similarity(np.atleast_2d(jd_embeddings), np.atleast_2d(resume_embeddings))
        skill_count = 0
        total_skills = len(jd_skills)

        for similarity in similarities:
            max_similar = np.max(similarity)
            if max_similar >= self.__skill_match_threshold:
                skill_count += 1

        return (skill_count / total_skills) * score
    

    # experience_years_match_score: sum resume experience durations (using start_year/end_year,
    # "Present" resolves to current year) into total_experience. Score = 0 if below
    # min_experience_yrs; assigned score if within [min_experience_yrs, max_experience_yrs];
    # assigned score minus an overage penalty if above max_experience_yrs, floored at 0.
    def get_experience_score(self, resume: Resume, job_description: JobDescription):
        total_experience: int = 0
        current_year = date.today().year
        year_pattern = re.compile(r"^\d{4}$")
        present_end_count = 0

        for experience in resume.experience:
            start_year = experience.start_year
            end_year = experience.end_year

            if start_year is None:
                continue

            if end_year is None:
                end_year = start_year

            if start_year == "Present":
                raise Exception("experience_score :: start_year cannot be 'Present'")

            if not year_pattern.match(start_year):
                raise Exception(f"experience_score :: Invalid start_year format: {start_year}")

            if end_year == "Present":
                present_end_count += 1
                if present_end_count > 1:
                    raise Exception("experience_score :: More than one experience has end_year as 'Present'")
                end = current_year
            elif year_pattern.match(end_year):
                end = int(end_year)
            else:
                raise Exception(f"experience_score :: Invalid end_year format: {end_year}")

            start = int(start_year)
            if start > end:
                raise Exception(f"experience_score :: start_year {start} is greater than end_year {end}")

            total_experience += end - start

        min_experience_yrs = job_description.min_experience_yrs
        max_experience_yrs = job_description.max_experience_yrs

        if not min_experience_yrs and not max_experience_yrs:
            return Constants.EXPERIENCE_YEARS_MATCH_SCORE
        
        if not min_experience_yrs or min_experience_yrs is None:
            min_experience_yrs = 0

        if not max_experience_yrs or max_experience_yrs is None:
            max_experience_yrs = 9999

        min_experience_yrs = float(min_experience_yrs)
        max_experience_yrs = float(max_experience_yrs)

        if total_experience < min_experience_yrs:
            return 0

        if total_experience <= max_experience_yrs:
            return Constants.EXPERIENCE_YEARS_MATCH_SCORE

        score = Constants.EXPERIENCE_YEARS_MATCH_SCORE - ((total_experience - max_experience_yrs) / Constants.EXPERIENCE_YEARS_MATCH_SCORE)
        return score if score > 0 else 0
    

    # education_match_score: match JD degree_needed embedding against each resume education.degree
    # embedding. Score = assigned EDUCATION_MATCH_SCORE if the max similarity clears
    # education_match_threshold, otherwise score = max similarity itself.
    def get_education_score(self, resume: Resume, job_description: JobDescription):
        jd_degree = job_description.degree_needed
        resume_degrees = [ education.degree if education.degree else "" for education in resume.education ]

        if not jd_degree:
            return Constants.EDUCATION_MATCH_SCORE
        
        if not len(resume_degrees):
            return 0
        
        jd_embeddings = np.asarray(self.__model.encode(jd_degree))
        resume_embeddings = np.asarray(self.__model.encode(resume_degrees))

        similarities = cosine_similarity(np.atleast_2d(jd_embeddings), np.atleast_2d(resume_embeddings))
        max_similar = np.max(similarities[0])

        return max_similar if max_similar*Constants.EDUCATION_MATCH_SCORE < self.__education_match_threshold else Constants.EDUCATION_MATCH_SCORE
    

    # certification_match_score: match JD certifications_required embeddings against resume
    # certification name embeddings, count certs whose max similarity exceeds
    # certifications_match_threshold, score = (matched cert count / total JD certs required)
    # * assigned CERTIFICATION_MATCH_SCORE.
    def get_certifications_score(self, resume: Resume, job_description: JobDescription):
        resume_certs = [ cert.name for cert in resume.certifications ]
        jd_certs = job_description.certifications_required
            
        if not len(jd_certs):
            return Constants.CERTIFICATION_MATCH_SCORE
        
        if not len(resume_certs):
            return 0
            
        jd_embeddings = np.asarray(self.__model.encode(jd_certs))
        resume_embeddings = np.asarray(self.__model.encode(resume_certs))

        similarities = cosine_similarity(np.atleast_2d(jd_embeddings), np.atleast_2d(resume_embeddings))
        cert_count = 0
        total_certs = len(jd_certs)

        for similarity in similarities:
            max_similar = np.max(similarity)
            if max_similar >= self.__certifications_match_threshold:
                cert_count += 1

        return (cert_count / total_certs) * Constants.CERTIFICATION_MATCH_SCORE
    

    # overall_score: sum of summary, role, all skill (required/general/soft), experience,
    # education, and certification match scores.
    def get_overall_score(self, resume: Resume, job_description: JobDescription):

        summary_score = self.get_summary_score(resume, job_description)
        role_score = self.get_role_score(resume, job_description)
        required_skill_score = self.get_skills_score(resume, job_description, "required")
        general_skill_score = self.get_skills_score(resume, job_description, "general")
        soft_skill_score = self.get_skills_score(resume, job_description, "soft")
        experience_score = self.get_experience_score(resume, job_description)
        education_score = self.get_education_score(resume, job_description)
        certification_score = self.get_certifications_score(resume, job_description)

        return summary_score + role_score + required_skill_score + general_skill_score + soft_skill_score + experience_score + education_score + certification_score

    
    def get_model(self):
        return self.__model
    
    def get_parameters(self):
        return {
            "certifications_penalty": self.__certifications_penalty,
            "projects_pentalty": self.__projects_pentalty,
            "skill_match_threshold": self.__skill_match_threshold,
            "education_match_threshold": self.__education_match_threshold,
            "certifications_match_threshold": self.__certifications_match_threshold
        }
    
    def get_model_name(self):
        return self.__model_name

    def get_device(self):
        return self.__device
    
    def tweak_parameter(self, certifications_penalty=None, projects_pentalty=None, skill_match_threshold=None, education_match_threshold=None, certifications_match_threshold=None):
        self.__certifications_penalty = certifications_penalty if certifications_penalty else self.__certifications_penalty
        self.__projects_pentalty = projects_pentalty if projects_pentalty else self.__projects_pentalty
        self.__skill_match_threshold = skill_match_threshold if skill_match_threshold else self.__skill_match_threshold
        self.__education_match_threshold = education_match_threshold if education_match_threshold else self.__education_match_threshold
        self.__certifications_match_threshold = certifications_match_threshold if certifications_match_threshold else self.__certifications_match_threshold
