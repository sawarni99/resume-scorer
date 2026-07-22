# Resume Scorer

An AI tool that scores a resume against a job description for IT / Computer Science professionals.

The pipeline extracts structured data from both a resume PDF and a job description, then produces a weighted match score (out of 100) built from semantic similarity between the two using vector embeddings.

## Demo

A quick walkthrough of the Streamlit app scoring a resume against a job description:

https://github.com/sawarni99/resume-scorer/raw/main/images/resume_scorer.mp4

> If the video does not play inline, [download / view it here](https://github.com/sawarni99/resume-scorer/raw/main/images/resume_scorer.mp4).

## Architecture

![Resume Scorer architecture](https://github.com/sawarni99/resume-scorer/raw/main/images/resume_scorer_architecture.png)

## How it works

1. **Extract text from the resume PDF** — resumes are computer-generated, so text is pulled directly with [PyMuPDF (`fitz`)](https://pymupdf.readthedocs.io/) rather than OCR.
2. **Convert the resume to JSON** — a fine-tuned LLM parses the raw text into a structured `Resume` object.
3. **Convert the job description to JSON** — a second fine-tuned LLM parses the JD text into a structured `JobDescription` object.
4. **Predefine weights** for each JD-derived section.
5. **Match and score** JD sections against resume sections using cosine similarity over vector embeddings.

## Resume sections

A resume is modeled as the following sections (see [datatypes/resume_data_type.py](datatypes/resume_data_type.py)):

1. Contact Info
2. Description
3. Education
4. Skills
5. Work Experience
6. Projects
7. Certifications

## Project layout

| Path | Description |
| --- | --- |
| [api/main.py](api/main.py) | FastAPI service exposing extraction and scoring endpoints |
| [datatypes/](datatypes/) | Pydantic models for `Resume` and `JobDescription` |
| [utils/scorer.py](utils/scorer.py) | Embedding-based scoring logic |
| [utils/constants.py](utils/constants.py) | Model names, prompts, weights, and thresholds |
| [utils/ollama_util.py](utils/ollama_util.py) | Ollama wrapper (pull / exists / chat) for running the fine-tuned GGUF models locally |
| [utils/resume_util.py](utils/resume_util.py), [utils/jd_util.py](utils/jd_util.py) | PDF text extraction, dataset analysis, JSON validation helpers |
| [notebooks/](notebooks/) | Dataset creation, fine-tuning, and scoring experiments |
| [datasets/](datasets/) | Generated CSV datasets |
| [images/](images/) | Training / evaluation loss curves |

## Datasets

600 resumes were generated with Claude (structured and unstructured layouts) covering Computer Science roles, plus 500 public-style job descriptions. All datasets are published on Hugging Face.

### Resume extraction dataset

Extracted resume text → structured JSON. Published as [`sawarni99/resume-to-json-dataset`](https://huggingface.co/datasets/sawarni99/resume-to-json-dataset).

**Creation steps:**

1. Prompt Claude to generate 600 Computer-Science resumes in both structured and unstructured styles.
2. Parse the generated PDFs into text with `fitz` (no OCR — they are computer-generated).
3. Save the extracted text to a CSV.
4. Prompt Claude to produce the JSON-formatted extraction for each.
5. Publish the `extracted text → JSON` dataset to Hugging Face.

**Resume JSON schema:**

```json
{
  "description": "",
  "education": [
    { "school_name": "", "year": "", "degree": "" }
  ],
  "skills": [],
  "experience": [
    { "company_name": "", "start_year": "", "end_year": "", "descriptions": [] }
  ],
  "projects": [
    { "name": "", "description": "" }
  ],
  "certifications": [
    { "name": "", "institute_name": "", "description": "" }
  ]
}
```

**Analysis results:**

- Total rows: **600**
- Invalid JSONs: **0**
- Inferred values (not found in resume text): **0**
- Max characters in resume text: **5,926**
- Max words in resume text: **851**
- Max tokens in resume text: **1,359**
- Max tokens in structured JSON: **1,178**
- System prompt tokens: **261**
- User prompt tokens: **15**

### Job description extraction dataset

Job description text → structured JSON. Published as [`sawarni99/jd-to-json-dataset`](https://huggingface.co/datasets/sawarni99/jd-to-json-dataset).

**Creation steps:**

1. Gather a generated dataset of public-style job descriptions.
2. Save them to a CSV.
3. Prompt Claude to produce the JSON-formatted extraction for each.
4. Publish the `JD text → JSON` dataset to Hugging Face.

**Job description JSON schema:**

```json
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
```

**Analysis results:**

- Total rows: **500**
- Invalid JSONs: **0**
- Inferred values (not found in JD text): **0**
- Non-numeric min/max experience years: **0**
- Max characters in JD text: **1,891**
- Max words in JD text: **275**
- Max tokens in JD text: **368**
- Max tokens in JD JSON: **388**
- System prompt tokens: **205**
- User prompt tokens: **16**

## Fine-tuning

Both extractors fine-tune the same base model: [`unsloth/Llama-3.2-3B-Instruct`](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct).

**Steps:**

1. Quantize the ~6 GB model down to ~3 GB with 4-bit quantization via [Unsloth](https://github.com/unslothai/unsloth).
2. Add LoRA adapters (rank 16) trained on the self-attention layers.
3. Export the model to GGUF format on Hugging Face so it can be served locally with [Ollama](https://ollama.com/).

**Published models:**

| | Resume extractor | JD extractor |
| --- | --- | --- |
| LoRA adapter | `sawarni99/resume-extraction-unsloth-llama-3.2-3B-Instruct-peft` | `sawarni99/jd-extraction-unsloth-llama-3.2-3B-Instruct-peft` |
| GGUF (Ollama) | `hf.co/sawarni99/resume-extraction-unsloth-llama-3.2-3B-Instruct-gguf.1` | `hf.co/sawarni99/jd-extraction-unsloth-llama-3.2-3B-Instruct-gguf` |

**Evaluation strategy:** ROUGE score, exact match, and valid-JSON rate.

**Training & evaluation loss curves:**

_Resume extraction_

| Training loss | Evaluation loss |
| --- | --- |
| ![Resume extraction training loss](https://github.com/sawarni99/resume-scorer/raw/main/images/resume_extraction_train_loss.png) | ![Resume extraction evaluation loss](https://github.com/sawarni99/resume-scorer/raw/main/images/resume_extraction_eval_loss.png) |

_Job description extraction_

| Training loss | Evaluation loss |
| --- | --- |
| ![JD extraction training loss](https://github.com/sawarni99/resume-scorer/raw/main/images/jd_extraction_train_loss.png) | ![JD extraction evaluation loss](https://github.com/sawarni99/resume-scorer/raw/main/images/jd_extraction_eval_loss.png) |

## Scoring

The overall score is out of **100**, distributed across the following metrics (weights in [utils/constants.py](utils/constants.py)):

| Metric | Weight |
| --- | --- |
| Summary match | 5 |
| Role match | 23 |
| **Skills match** | **30** |
| &nbsp;&nbsp;&nbsp;&nbsp;Required skills | 20 |
| &nbsp;&nbsp;&nbsp;&nbsp;General skills | 7 |
| &nbsp;&nbsp;&nbsp;&nbsp;Soft skills | 3 |
| Experience years match | 22 |
| Education match | 8 |
| Certification match | 12 |
| **Overall** | **100** |

**Embedder model:** [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (auto-detects CUDA / MPS / CPU).

### Scoring rules

Each score is implemented in [utils/scorer.py](utils/scorer.py). If a JD section is empty, the resume is given full credit for that section; if the JD requires something the resume lacks, that section scores 0.

**`summary_match_score`** — resume `description` vs JD `job_summary`.
`score = cosine_similarity(resume, jd) * assigned_score`

**`role_match_score`** — JD `role_description` vs resume `experience.descriptions`, `projects.description`, `certification.description`.
1. Match each JD role embedding against all resume description embeddings.
2. Take the max cosine similarity per role, applying a penalty to projects (`0.8`) and certifications (`0.85`) matches.
3. `score = (sum of max similarities / number of role_descriptions) * assigned_score`

**`skills_match_score`** — resume `skills` vs JD `required_skills` / `general_skills` / `soft_skills` (scored separately).
1. Match each JD skill embedding against all resume skill embeddings.
2. Count skills whose max similarity is `>= 0.55`.
3. `score = (matched skills / total JD skills of this type) * assigned_score`

**`experience_years_match_score`** — computed from `start_year` / `end_year` (`Present` resolves to the current year).
- `0` if `total_experience < min_experience_yrs`
- full `assigned_score` if `min <= total_experience <= max`
- `assigned_score - ((total_experience - max) / assigned_score)` if above max, floored at `0`

**`education_match_score`** — JD `degree_needed` vs resume `education.degree`.
`score = assigned_score` if the max similarity clears the education threshold (`0.5705`), otherwise the max similarity itself.

**`certification_match_score`** — JD `certifications_required` vs resume `certifications.name`.
1. Match each JD cert embedding against all resume cert embeddings.
2. Count certs whose max similarity is `>= 0.605`.
3. `score = (matched certs / total required certs) * assigned_score`

**`overall_score`** — the sum of all of the above.

> Thresholds and penalties (`skill_match_threshold`, `education_match_threshold`, `certifications_match_threshold`, `projects_penalty`, `certifications_penalty`) are tunable at runtime via `Scorer.tweak_parameter(...)`.

## API

The [FastAPI](https://fastapi.tiangolo.com/) service ([api/main.py](api/main.py)) loads the embedding model once at startup and exposes:

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/resume/extract` | Upload a resume PDF → structured `Resume` JSON |
| `POST` | `/job-description/extract` | Send JD text → structured `JobDescription` JSON |
| `POST` | `/score` | Send a `Resume` + `JobDescription` → per-section and overall scores |

The extraction endpoints run the fine-tuned GGUF models through Ollama, pulling them automatically on first use, and retry up to 3 times on invalid JSON.

## Streamlit app

A [Streamlit](https://streamlit.io/) front-end ([streamlit_app.py](streamlit_app.py)) provides a UI over the API: upload a resume PDF, paste a job description, and get a weighted score breakdown with a per-section bar chart and a match verdict (Strong / Moderate / Weak). It also exposes tabs for the parsed resume and job description JSON, and lets you download the full report.

The app talks to the FastAPI service, so that must be running first. See the [Demo](#demo) above for a walkthrough.

## Getting started

This project uses [`uv`](https://docs.astral.sh/uv/) (Python `>= 3.12`).

```bash
# Install dependencies
uv sync

# Make sure Ollama is running locally for the extraction endpoints
# https://ollama.com/download

# Run the API (in one terminal)
uv run uvicorn api.main:app --reload

# Run the Streamlit app (in another terminal)
uv run streamlit run streamlit_app.py
```

Then open the Streamlit app in your browser (it launches at `http://localhost:8501`), or explore the API directly via the interactive docs at `http://127.0.0.1:8000/docs`.

### Example: score a resume

```bash
# 1. Extract structured data from a resume PDF
curl -X POST http://127.0.0.1:8000/resume/extract \
  -F "file=@resumes/resume_006.pdf"

# 2. Extract structured data from a job description
curl -X POST http://127.0.0.1:8000/job-description/extract \
  -H "Content-Type: application/json" \
  -d '{"job_description_text": "We are hiring a backend engineer ..."}'

# 3. Score the resume against the job description
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{"resume": { ... }, "job_description": { ... }}'
```

## Notebooks

The [notebooks/](notebooks/) directory documents the end-to-end workflow:

- `resume_pdf_to_text_creation.ipynb` — parse resume PDFs into text and build the CSV
- `resume_extraction.ipynb` — fine-tune and evaluate the resume extractor
- `JD_extraction.ipynb` — fine-tune and evaluate the JD extractor
- `resume_scoring.ipynb` — develop and tune the scoring logic
