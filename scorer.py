import json
import re

from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def score_project(extracted_data: dict, client: genai.Client) -> dict:
    prompt = f"""You are a senior development project evaluator.

Based on the extracted project information below, score the project on four dimensions. \
Each score must be an integer from 0 to 100.

Project information:
{json.dumps(extracted_data, indent=2)}

Scoring criteria:
- delivery   (0–100): How well is the project being implemented against its plan? \
Are milestones being met, activities completed on schedule?
- impact     (0–100): To what extent is the project achieving intended outcomes \
and beneficiary targets?
- risk_level (0–100): How well are risks being managed? \
100 = very low risk / well-managed; 0 = critical unresolved risks.
- efficiency (0–100): Is the project making good use of budget, staff, and time?

Return ONLY a valid JSON object — no markdown, no explanation:
{{
  "delivery": {{
    "score": 75,
    "justification": "One concise sentence explaining the score based on the document."
  }},
  "impact": {{
    "score": 65,
    "justification": "One concise sentence explaining the score based on the document."
  }},
  "risk_level": {{
    "score": 55,
    "justification": "One concise sentence explaining the score based on the document."
  }},
  "efficiency": {{
    "score": 70,
    "justification": "One concise sentence explaining the score based on the document."
  }}
}}

Base every score strictly on the document content. If information is limited, \
use a conservative mid-range score and note the limitation in the justification."""

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
