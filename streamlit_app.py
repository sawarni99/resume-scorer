"""Streamlit front-end for Resume Scorer.

Talks to the FastAPI service (api/main.py), which must be running:

    uv run uvicorn api.main:app --reload

Run this app with:

    uv run streamlit run streamlit_app.py
"""

import json

import requests
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"

# Section weights mirror utils/constants.py — used only for display.
SCORE_SECTIONS = [
    ("role_score", "Role match", 23),
    ("experience_score", "Experience years", 22),
    ("required_skill_score", "Required skills", 20),
    ("certification_score", "Certifications", 12),
    ("education_score", "Education", 8),
    ("general_skill_score", "General skills", 7),
    ("summary_score", "Summary match", 5),
    ("soft_skill_score", "Soft skills", 3),
]

VERDICTS = [
    (70, "Strong match", "good"),
    (45, "Moderate match", "warning"),
    (0, "Weak match", "critical"),
]


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class ApiError(Exception):
    pass


def _detail(response: requests.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def _post(api_url: str, path: str, **kwargs) -> dict:
    try:
        response = requests.post(f"{api_url.rstrip('/')}{path}", timeout=600, **kwargs)
    except requests.ConnectionError:
        raise ApiError(
            f"Could not reach the API at `{api_url}`. "
            "Start it with `uv run uvicorn api.main:app --reload` "
            "and make sure Ollama is running."
        )
    except requests.Timeout:
        raise ApiError("The API request timed out — the extraction model may still be downloading.")
    if response.status_code != 200:
        raise ApiError(_detail(response))
    return response.json()


def extract_resume(api_url: str, file_name: str, file_bytes: bytes) -> dict:
    return _post(api_url, "/resume/extract", files={"file": (file_name, file_bytes, "application/pdf")})


def extract_jd(api_url: str, jd_text: str) -> dict:
    return _post(api_url, "/job-description/extract", json={"job_description_text": jd_text})


def score(api_url: str, resume: dict, jd: dict) -> dict:
    return _post(api_url, "/score", json={"resume": resume, "job_description": jd})


# ---------------------------------------------------------------------------
# Score report rendering
# ---------------------------------------------------------------------------

REPORT_CSS = """
<style>
.rs-report {
  color-scheme: light;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --track: #e1e0d9;
  --border: rgba(11, 11, 11, 0.10);
  --bar: #2a78d6;
  --good: #0ca30c;
  --good-ink: #006300;
  --warning: #fab219;
  --critical: #d03b3b;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) {
  .rs-report {
    color-scheme: dark;
    --surface: #1a1a19;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --track: #2c2c2a;
    --border: rgba(255, 255, 255, 0.10);
    --bar: #3987e5;
    --good-ink: #0ca30c;
  }
}
.rs-report .rs-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px 28px;
}
.rs-hero { display: flex; align-items: center; gap: 24px; flex-wrap: wrap; margin-bottom: 20px; }
.rs-hero-number { font-size: 64px; font-weight: 700; line-height: 1; color: var(--ink); }
.rs-hero-denom { font-size: 22px; font-weight: 400; color: var(--muted); }
.rs-hero-meta { display: flex; flex-direction: column; gap: 6px; }
.rs-verdict {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 15px; font-weight: 600; color: var(--ink);
}
.rs-verdict-dot { width: 12px; height: 12px; border-radius: 50%; }
.rs-hero-sub { font-size: 13px; color: var(--ink-2); }
.rs-row { display: grid; grid-template-columns: 150px 1fr 76px; gap: 12px; align-items: center; padding: 5px 0; }
.rs-row-label { font-size: 13px; color: var(--ink-2); }
.rs-track { height: 10px; border-radius: 4px; background: var(--track); overflow: hidden; }
.rs-fill { height: 100%; border-radius: 4px; background: var(--bar); }
.rs-value { font-size: 13px; color: var(--ink); text-align: right; font-variant-numeric: tabular-nums; }
.rs-value .rs-max { color: var(--muted); }
</style>
"""


def verdict_for(overall: float) -> tuple[str, str]:
    for threshold, label, color_role in VERDICTS:
        if overall >= threshold:
            return label, color_role
    return VERDICTS[-1][1], VERDICTS[-1][2]


def render_report(scores: dict) -> str:
    overall = scores["overall_score"]
    label, color_role = verdict_for(overall)

    rows = []
    for key, name, weight in SCORE_SECTIONS:
        value = scores[key]
        pct = max(0.0, min(1.0, value / weight)) * 100
        rows.append(
            f'<div class="rs-row" title="{name}: {value:.1f} of {weight} points">'
            f'<div class="rs-row-label">{name}</div>'
            f'<div class="rs-track"><div class="rs-fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="rs-value">{value:.1f} <span class="rs-max">/ {weight}</span></div>'
            f"</div>"
        )

    return f"""{REPORT_CSS}
<div class="rs-report">
  <div class="rs-card">
    <div class="rs-hero">
      <div class="rs-hero-number">{overall:.1f}<span class="rs-hero-denom"> / 100</span></div>
      <div class="rs-hero-meta">
        <div class="rs-verdict">
          <span class="rs-verdict-dot" style="background: var(--{color_role})"></span>
          {label}
        </div>
        <div class="rs-hero-sub">Weighted semantic similarity across {len(SCORE_SECTIONS)} sections</div>
      </div>
    </div>
    {''.join(rows)}
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Resume Scorer", page_icon="🎯", layout="wide")

if "results" not in st.session_state:
    st.session_state.results = None

with st.sidebar:
    st.title("🎯 Resume Scorer")
    st.caption("Score a resume against a job description using fine-tuned extraction models and embedding similarity.")

    api_url = st.text_input("API endpoint", value=DEFAULT_API_URL, help="Base URL of the FastAPI service (api/main.py).")

    st.divider()
    st.markdown("**How the score is weighted**")
    for _, name, weight in SCORE_SECTIONS:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;font-size:13px;padding:2px 0;">'
            f"<span>{name}</span><span><b>{weight}</b></span></div>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption(
        "Pipeline: PDF text extraction (PyMuPDF) → fine-tuned Llama-3.2-3B extractors (Ollama) "
        "→ Qwen3 embeddings → weighted cosine-similarity score."
    )

st.header("Score a resume against a job description")

input_col, jd_col = st.columns(2, gap="large")

with input_col:
    st.subheader("1 · Resume")
    uploaded = st.file_uploader("Upload a resume (PDF)", type=["pdf"], accept_multiple_files=False)

with jd_col:
    st.subheader("2 · Job description")
    jd_text = st.text_area(
        "Paste the job description text",
        height=220,
        placeholder="We are hiring a backend engineer with 3+ years of experience in Python, ...",
    )

if uploaded is not None:
    resume_name, resume_bytes = uploaded.name, uploaded.getvalue()
else:
    resume_name, resume_bytes = None, None

ready = resume_name is not None and resume_bytes is not None and bool(jd_text.strip())
if st.button("Score resume", type="primary", disabled=not ready,
             help=None if ready else "Provide a resume PDF and a job description first.") \
        and resume_name is not None and resume_bytes is not None:
    try:
        with st.status("Scoring…", expanded=True) as status:
            st.write(f"Extracting structured data from **{resume_name}**…")
            resume_data = extract_resume(api_url, resume_name, resume_bytes)
            st.write("Extracting structured data from the job description…")
            jd_data = extract_jd(api_url, jd_text)
            st.write("Computing embedding-similarity scores…")
            scores = score(api_url, resume_data, jd_data)
            status.update(label="Done", state="complete", expanded=False)
        st.session_state.results = {
            "resume_name": resume_name,
            "resume": resume_data,
            "jd": jd_data,
            "scores": scores,
        }
    except ApiError as e:
        st.session_state.results = None
        st.error(str(e))

results = st.session_state.results
if results:
    st.divider()
    breakdown_tab, resume_tab, jd_tab = st.tabs(
        ["📊 Score breakdown", "📄 Parsed resume", "📋 Parsed job description"]
    )
    with breakdown_tab:
        st.markdown(render_report(results["scores"]), unsafe_allow_html=True)
        st.caption(
            f"Scored **{results['resume_name']}**. Empty JD sections get full credit; "
            "requirements missing from the resume score 0."
        )
        st.download_button(
            "Download full report (JSON)",
            data=json.dumps(results, indent=2),
            file_name="resume_score_report.json",
            mime="application/json",
        )
    with resume_tab:
        st.json(results["resume"])
    with jd_tab:
        st.json(results["jd"])
else:
    st.info("Upload a resume and paste a job description, then hit **Score resume**.")
