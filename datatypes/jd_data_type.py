from pydantic import BaseModel
from typing import List, Optional


class JobDescription(BaseModel):
    job_summary: Optional[str] = None
    role_description: List[str] = []
    required_skills: List[str] = []
    general_skills: List[str] = []
    soft_skills: List[str] = []
    min_experience_yrs: Optional[str] = None
    max_experience_yrs: Optional[str] = None
    degree_needed: Optional[str] = None
    certifications_required: List[str] = []
