import matplotlib.pyplot as plt
import numpy as np
from datatypes.jd_data_type import JobDescription
from utils.utility import Utility
from typing import List, Dict

class JobDescriptionUtil:

    @staticmethod
    def plot_jd_data(data, title: str = "", ylabel: str = "", xlabel: str = "") -> None:
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
    def get_inferred_texts_from_jd(jd: JobDescription, jd_text: str) -> List:
        util = Utility()
        inferred_texts = []
        normalized_text = util.normalize_strings_with_newlines(jd_text).lower()
        for value, is_date in util.leaf_strings_from_jd(jd):
            if is_date:
                found = util.date_in_text(value, normalized_text)
            else:
                found = util.normalize_strings_with_newlines(value).lower() in normalized_text
            if not found:
                inferred_texts.append(value)

        return inferred_texts
    
    @staticmethod
    def valid(jd_json:str) -> bool:
        try:
            JobDescription.model_validate_json(jd_json)
            return True
        except Exception as e:
            return False
        
    @staticmethod
    def create(jd_json:str) -> JobDescription:
        try:
            resume = JobDescription.model_validate_json(jd_json)
            return resume
        except Exception as e:
            raise ValueError(f"Invalid resume JSON: {e}")

    @staticmethod
    def format(jd: JobDescription) -> None:
        print(jd.model_dump_json(indent=4))

    @staticmethod
    def min_exp_is_number_or_none(jd: JobDescription):
        min_exp_yrs = jd.min_experience_yrs
        if not min_exp_yrs:
            return True
        try:
            float(min_exp_yrs)
            return True
        except Exception as e:
            return False
    
    @staticmethod
    def max_exp_is_number_or_none(jd: JobDescription):
        max_exp_yrs = jd.max_experience_yrs
        if not max_exp_yrs:
            return True
        try:
            float(max_exp_yrs)
            return True
        except Exception as e:
            return False