import io
import json
import re

import fitz  # PyMuPDF
from docx import Document
from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"


def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace").strip()


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Dispatch to the correct extractor based on file extension."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    if ext == ".docx":
        return extract_text_from_docx(file_bytes)
    if ext == ".txt":
        return extract_text_from_txt(file_bytes)
    raise ValueError(
        f"Unsupported file type '{ext}'. Please upload a PDF, Word (.docx), or plain text (.txt) file."
    )


def _clean_json(raw: str) -> str:
    """Strip markdown fences and surrounding whitespace."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def extract_project_data(text: str, client: genai.Client) -> dict:
    prompt = f"""You are an expert project analyst. \
Analyse the following project document. It may be a project report, evaluation document, \
programme assessment, annual report, or M&E document.

Extract the information below and return ONLY a valid JSON object — no markdown, \
no explanation, no surrounding text.

Required JSON structure:
{{
  "project_name": "Full project name, or 'Unknown' if not found",
  "funder": "Primary funding organisation (e.g. World Bank, GIZ, USAID, Federal Government of Nigeria), or 'Unknown'",
  "implementing_organization": "Lead implementing organisation, or 'Unknown'",
  "project_status": "One of: On Track | At Risk | Completed | Unknown",
  "objectives": ["Objective 1", "Objective 2", "..."],
  "achievements": ["Achievement 1", "Achievement 2", "..."],
  "challenges": ["Challenge or risk 1", "Challenge or risk 2", "..."],
  "recommendations": ["Recommendation 1", "Recommendation 2", "..."]
}}

Rules:
- Each list should have 3–8 items where possible.
- Infer project_status from language cues (e.g. "on schedule", "delayed", "completed", "at risk").
- If a section has no clear content, return an empty list [].
- Do NOT invent information not present in the document.

Document text:
\"\"\"
{text[:14000]}
\"\"\"

Return only the JSON object."""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.3),
    )
    raw = _clean_json(response.text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Gemini returned non-JSON output:\n{raw[:500]}")
