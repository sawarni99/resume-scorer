
class Constants:

    # Resume Extraction Constants...
    EXTRACTION_DATASET_NAME = "sawarni99/resume-to-json-dataset"
    EXTRACTION_BASE_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
    EXTRACTION_FT_ADAPTER_NAME = "sawarni99/resume-extraction-unsloth-llama-3.2-3B-Instruct-peft"
    EXTRACTION_FT_GGUF_NAME = "resume-extraction-unsloth-llama-3.2-3B-Instruct-gguf.1"
    EXTRACTION_FT_OLLAMA_ENDPOINT = f"hf.co/sawarni99/{EXTRACTION_FT_GGUF_NAME}"
    EXTRACTION_SYSTEM_PROMPT_TEMPLATE = '''You are a resume parser. Extract information from the resume text provided and return it as a single valid JSON object.
OUTPUT RULES:
- Output ONLY the JSON object. No markdown fences, no explanations, no commentary before or after.
- Use exactly these keys, in this order, every time.
- For any missing string value, use an empty string "". For any missing array, use an empty array []. Never use null and never omit a key.
- All years (year, start_year, end_year) should be in the format YYYY or "Present" if currently employed or studying.
- Do not invent or infer any information. Only extract what is present in the resume text.

SCHEMA:
{
  "description": "",
  "education": [
    {
      "school_name": "",
      "year": "",
      "degree": ""
    }
  ],
  "skills": [],
  "experience": [
    {
      "company_name": "",
      "start_year": "",
      "end_year": "",
      "descriptions": []
    }
  ],
  "projects": [
    {
      "name": "",
      "description": ""
    }
  ],
  "certifications": [
    {
      "name": "",
      "institute_name": "",
      "description": ""
    }
  ]
}
'''
    EXTRACTION_USER_PROMPT_TEMPLATE = '''Extract the structured data from the following resume: 
<resume_text>
'''

    # Job Description Extraction
    JD_EXTRACTION_DATASET_NAME = "sawarni99/jd-to-json-dataset"
    JD_EXTRACTION_BASE_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
    JD_EXTRACTION_FT_ADAPTER_NAME = "sawarni99/jd-extraction-unsloth-llama-3.2-3B-Instruct-peft"
    JD_EXTRACTION_FT_GGUF_NAME = "jd-extraction-unsloth-llama-3.2-3B-Instruct-gguf"
    JD_EXTRACTION_FT_OLLAMA_ENDPOINT = f"hf.co/sawarni99/{JD_EXTRACTION_FT_GGUF_NAME}"
    JD_EXTRACTION_SYSTEM_PROMPT = """You are a Job Description parser. Extract information from the given job description and return a valid a single valid JSON.

OUTPUT RULES:
- Output ONLY the JSON object. No markdown fences, no explanations, no commentary before or after.
- Use exactly these keys, in this order, every time.
- For any missing string value, use an empty string "". For any missing array, use an empty array []. Never use null and never omit a key.
- Do not invent or infer any information. Only extract what is present in the job description text.
- min_experience_yrs should be a number parsed as a string.
- Leave max_experience_yrs empty if there is no maximum experience required.

SCHEMA:
{
    "job_summary": "",
    "role_description": [],
    "required_skills": [],
    "general_skills": [],
    "soft_skills": [],
    "min_experience_yrs": "",
    "max_experience_yrs": "",
    "degree_needed": "",
    "certifications_required": []
}
"""
    JD_EXTRACTION_USER_PROMPT_TEMPLATE = '''Extract the structured data from the following job description: 
<jd_text>
'''
