from pydantic import BaseModel
from typing import List, Optional


class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None


class Education(BaseModel):
    school_name: Optional[str] = None
    year: Optional[str] = None
    degree: Optional[str] = None


class Experience(BaseModel):
    company_name: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    descriptions: List[str] = []


class Project(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class Certification(BaseModel):
    name: Optional[str] = None
    institute_name: Optional[str] = None
    description: Optional[str] = None


class Resume(BaseModel):
    # contact_info: ContactInfo
    description: Optional[str] = None
    education: List[Education] = []
    skills: List[str] = []
    experience: List[Experience] = []
    projects: List[Project] = []
    certifications: List[Certification] = []
