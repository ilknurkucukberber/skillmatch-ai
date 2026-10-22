# SkillMatch AI

A bilingual resume and job matching app built with **Streamlit**, **Sentence Transformers**, **scikit-learn**, and **PyMuPDF**. It keeps the core multilingual semantic matching workflow while improving the resume-review experience with an evidence-based CV improvement advisor, better Turkish/English UX, and a safer, more transparent recommendation model.

## Features
- Upload a PDF/TXT resume and paste a job description
- Calculate multilingual semantic similarity with a cached SentenceTransformer model
- Detect technical skills in Turkish and English with normalization for common aliases
- Show matched vs missing skills with a clear bar chart
- Export a detailed JSON report with analysis metadata and resume improvement recommendations
- Switch between Turkish and English without re-uploading the CV or rerunning analysis; analysis scores and detected skills remain unchanged
- Replace the old skill-by-skill learning list with a structured CV improvement advisor that reviews summary quality, project clarity, skills emphasis, unnecessary content, structure, and ATS readability
- Keep the app honest: no hiring guarantee or fabricated credentials

## Resume improvement advisor
The current advisor section is intentionally deterministic and evidence-based. It does not claim to be a generative AI system. Instead, it:

- inspects the actual resume structure and headings,
- checks for missing or weak professional summaries,
- identifies project or skill evidence that should be emphasized,
- flags generic or distracting content only when it is actually present,
- suggests clearer ordering and stronger ATS-friendly formatting,
- keeps suggestions grounded in the uploaded resume and the target job description.

The advisor stores recommendation types, priorities, and verbatim evidence separately from their display text. Its descriptions and suggested wording are rendered in the selected interface language and avoid asserting facts that are not present in the CV. Original resume and job-description excerpts remain unchanged.

JSON report labels and generated explanatory text use the language selected when the report is downloaded. Numerical scores, skill identifiers, and analysis metadata are language-independent.

## Important notes
- This is a portfolio/educational project, not a hiring decision system.
- Semantic similarity is not a probability of employment.
- Image-only or scanned PDFs require OCR and are not supported as searchable text.
- CV data is processed only in memory for the current running session; no uploaded content is intentionally persisted.

## Run locally
Python 3.11 or 3.12 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first run downloads the multilingual Hugging Face model and may take a few minutes depending on network speed. Subsequent analyses are much faster because the model is cached per process.

## Deploy
Push the repository to GitHub, then use the Streamlit Community Cloud deployment flow to start the app from `app.py`. Keep the first boot in mind if model download is needed. For public demos, avoid uploading personal or confidential CV data.

## Limitations and next steps
- Skill extraction remains keyword- and alias-based, so it can miss nuanced wording or advanced synonyms.
- Resume quality, missing context, and real-world experience still require human interpretation.
- The recommendations are algorithmic and evidence-driven, not LLM-generated claims.
- A future optional local model could enhance explanation quality, but the production feature works without API keys and without a large inference dependency.
