import json

from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"


def generate_stakeholder_report(
    extracted_data: dict, scores: dict, client: genai.Client
) -> str:
    prompt = f"""You are a senior technical writer specialising in project management and advisory.

Write a professional one-page stakeholder report based on the project data below. \
This report will be shared with donors, government counterparts, and senior management.

Project Data:
{json.dumps(extracted_data, indent=2)}

Project Health Scores:
- Delivery:    {scores['delivery']['score']}/100 — {scores['delivery']['justification']}
- Impact:      {scores['impact']['score']}/100 — {scores['impact']['justification']}
- Risk Level:  {scores['risk_level']['score']}/100 — {scores['risk_level']['justification']}
- Efficiency:  {scores['efficiency']['score']}/100 — {scores['efficiency']['justification']}

Structure the report with these exact section headings (use ALL CAPS for headings):

EXECUTIVE SUMMARY
KEY FINDINGS
RISK ASSESSMENT
RECOMMENDATIONS

Formatting requirements:
- Begin with the project name and "Reporting Period: Q1 2025" on the first two lines.
- Write in formal development-sector English (World Bank / GIZ report style).
- Each section should be 3–5 sentences or concise bullet points.
- Be specific and factual — reference actual data from the project information.
- Do not include JSON, markdown, or technical formatting.
- End with: "Prepared by DocuSense | Document Intelligence System"

Write the report now:"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7),
    )
    return response.text.strip()
