
import re
from datatypes.resume_data_type import Resume
from typing import List


class Utility:

    @staticmethod
    def remove_special_chars(text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9\s/,.:;'\"~!@#$%^&*()\[\]{}\\\|_+\-]", "", text)


    def normalize_strings_with_newlines(self, string: str) -> str:
        """Replace newlines (and any surrounding whitespace) with a single space."""
        return " ".join(string.split())
    

    def date_in_resume_text(self, date_value: str, normalized_text: str) -> bool:
        """Check if a start_year/end_year value can be found in the resume text."""
        
        val = date_value.strip().lower()

        # "Present" / "Current" are always valid — they signal an ongoing role
        if val in ("present", "current"):
            return True
        if self.normalize_strings_with_newlines(date_value).lower() in normalized_text:
            return True
        
        # For 4-digit years like "2026", also accept 2-digit shorthand e.g. '26 or standalone 26
        if len(val) == 4 and val.isdigit():
            short = val[2:]
            if re.search(rf"'{short}|\b{short}\b", normalized_text):
                return True
        return False
    

    def leaf_strings(self, resume: Resume) -> List[tuple[str, bool]]:
        """Extract every non-empty string leaf value from a Resume. Returns (value, is_date_field) tuples."""
        values = []

        if resume.description and resume.description.strip():
            values.append((resume.description, False))

        for edu in resume.education:
            for v in [edu.school_name, edu.year, edu.degree]:
                if v and v.strip():
                    values.append((v, False))

        for skill in resume.skills:
            if skill and skill.strip():
                values.append((skill, False))

        for exp in resume.experience:
            if exp.company_name and exp.company_name.strip():
                values.append((exp.company_name, False))
            for date_v in [exp.start_year, exp.end_year]:
                if date_v and date_v.strip():
                    values.append((date_v, True))
            for desc in exp.descriptions:
                if desc and desc.strip():
                    values.append((desc, False))

        for proj in resume.projects:
            for v in [proj.name, proj.description]:
                if v and v.strip():
                    values.append((v, False))

        for cert in resume.certifications:
            for v in [cert.name, cert.institute_name, cert.description]:
                if v and v.strip():
                    values.append((v, False))

        return values
    
    @staticmethod
    def write_to_file(path: str, content: str):
        with open(path, 'w') as f:
                f.write(content)
        
    