import os
import tempfile
from contextlib import asynccontextmanager
from typing import Callable, Dict, List, TypeVar

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from datatypes.jd_data_type import JobDescription
from datatypes.resume_data_type import Resume
from utils.constants import Constants
from utils.jd_util import JobDescriptionUtil
from utils.ollama_util import Ollama
from utils.resume_util import ResumeUtil
from utils.scorer import Scorer


num_retry = 3

@asynccontextmanager
async def lifespan(app: FastAPI):
    # The Scorer loads a sentence-transformer embedding model, which is
    # expensive, so it's created once at startup and reused across requests.
    app.state.scorer = Scorer(Constants.SCORER_MODEL_NAME)
    yield


app = FastAPI(lifespan=lifespan)


class JobDescriptionExtractRequest(BaseModel):
    job_description_text: str


class ScoreRequest(BaseModel):
    resume: Resume
    job_description: JobDescription


class ScoreResponse(BaseModel):
    overall_score: float
    summary_score: float
    role_score: float
    required_skill_score: float
    general_skill_score: float
    soft_skill_score: float
    experience_score: float
    education_score: float
    certification_score: float


T = TypeVar("T")


def _run_ollama_extraction(ollama_endpoint: str, messages: List[Dict[str, str]]) -> str:
    ollama = Ollama(ollama_endpoint)
    try:
        if not ollama.exists():
            ollama.pull()
        response = ollama.chat(messages)
        if response is None:
            raise HTTPException(status_code=502, detail="No response from the extraction model.")
        return response["message"]["content"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Extraction model error: {str(e)}")


def _extract_with_retry(ollama_endpoint: str, messages: List[Dict[str, str]], parse: Callable[[str], T]) -> T:
    last_error: ValueError | None = None
    for _ in range(num_retry):
        structured_json = _run_ollama_extraction(ollama_endpoint, messages)
        try:
            return parse(structured_json)
        except ValueError as e:
            last_error = e
    raise HTTPException(
        status_code=502,
        detail=f"Model returned invalid JSON after {num_retry} attempts: {last_error}",
    )


@app.post("/resume/extract", response_model=Resume)
async def extract_resume(file: UploadFile = File(...)) -> Resume:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        resume_text = ResumeUtil.extract_text(tmp_path)
        if not resume_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract any text from the provided PDF.")

        messages = [
            {"role": "system", "content": Constants.EXTRACTION_SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": Constants.EXTRACTION_USER_PROMPT_TEMPLATE.replace("<resume_text>", resume_text)},
        ]
        return _extract_with_retry(Constants.EXTRACTION_FT_OLLAMA_ENDPOINT, messages, ResumeUtil.create)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/job-description/extract", response_model=JobDescription)
async def extract_job_description(request: JobDescriptionExtractRequest) -> JobDescription:
    jd_text = request.job_description_text
    if not jd_text or not jd_text.strip():
        raise HTTPException(status_code=400, detail="job_description_text cannot be empty.")

    messages = [
        {"role": "system", "content": Constants.JD_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": Constants.JD_EXTRACTION_USER_PROMPT_TEMPLATE.replace("<jd_text>", jd_text)},
    ]
    return _extract_with_retry(Constants.JD_EXTRACTION_FT_OLLAMA_ENDPOINT, messages, JobDescriptionUtil.create)


@app.post("/score", response_model=ScoreResponse)
async def score_resume(request: ScoreRequest) -> ScoreResponse:
    scorer: Scorer = app.state.scorer
    resume = request.resume
    job_description = request.job_description

    try:
        summary_score = scorer.get_summary_score(resume, job_description)
        role_score = scorer.get_role_score(resume, job_description)
        required_skill_score = scorer.get_skills_score(resume, job_description, "required")
        general_skill_score = scorer.get_skills_score(resume, job_description, "general")
        soft_skill_score = scorer.get_skills_score(resume, job_description, "soft")
        experience_score = scorer.get_experience_score(resume, job_description)
        education_score = scorer.get_education_score(resume, job_description)
        certification_score = scorer.get_certifications_score(resume, job_description)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Unable to score resume: {str(e)}")

    overall_score = (
        summary_score
        + role_score
        + required_skill_score
        + general_skill_score
        + soft_skill_score
        + experience_score
        + education_score
        + certification_score
    )

    return ScoreResponse(
        overall_score=float(overall_score),
        summary_score=float(summary_score),
        role_score=float(role_score),
        required_skill_score=float(required_skill_score),
        general_skill_score=float(general_skill_score),
        soft_skill_score=float(soft_skill_score),
        experience_score=float(experience_score),
        education_score=float(education_score),
        certification_score=float(certification_score),
    )
