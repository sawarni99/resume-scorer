import matplotlib.pyplot as plt
import numpy as np
from datatypes.resume_data_type import Resume
from utils.utility import Utility
from typing import List, Dict
import fitz


class ResumeUtil:

    @staticmethod
    def plot_resume_data(data, title: str = "", ylabel: str = "", xlabel: str = "") -> None:
        plt.figure(figsize=(7, 5))

        plt.hist(data, bins=30, color="steelblue", edgecolor="white")
        plt.axvline(np.mean(data), color="red", linestyle="--", label=f"Mean: {np.mean(data):.0f}")
        plt.axvline(np.median(data), color="orange", linestyle="--", label=f"Median: {np.median(data):.0f}")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"Min:    {min(data):>6}")
        print(f"Max:    {max(data):>6}")
        print(f"Mean:   {np.mean(data):>6.0f}")
        print(f"Median: {np.median(data):>6.0f}")
        print(f"Std:    {np.std(data):>6.0f}")


    @staticmethod
    def get_inferred_texts_from_resume(resume: Resume, resume_text:str) -> List:
        util = Utility()
        inferred_texts = []
        normalized_text = util.normalize_strings_with_newlines(resume_text).lower()
        for value, is_date in util.leaf_strings(resume):
            if is_date:
                found = util.date_in_resume_text(value, normalized_text)
            else:
                found = util.normalize_strings_with_newlines(value).lower() in normalized_text
            if not found:
                inferred_texts.append(value)

        return inferred_texts
    
    @staticmethod
    def valid(resume_json:str) -> bool:
        try:
            Resume.model_validate_json(resume_json)
            return True
        except Exception as e:
            return False
        
    @staticmethod
    def create(resume_json:str) -> Resume:
        try:
            resume = Resume.model_validate_json(resume_json)
            return resume
        except Exception as e:
            raise ValueError(f"Invalid resume JSON: {e}")

    @staticmethod
    def format(resume: Resume) -> None:
        print(resume.model_dump_json(indent=4))

    @staticmethod
    def extract_text(file_path: str) -> str:

        # Extrating data...
        doc = fitz.open(file_path)
        result:str = ""

        # Joining all the extracted pages in one text...
        for page in doc:
            text = "".join(page.get_text("text"))
            result += text

        # Sanitizing the data...
        file_text:str = Utility.remove_special_chars(result)

        return file_text
    

    