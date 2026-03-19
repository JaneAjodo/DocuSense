from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"


def answer_question(question: str, document_text: str, client: genai.Client) -> str:
    prompt = f"""You are an AI project analyst assistant. \
You have been given a project document to analyse.

Answer the question below using ONLY information found in the document. \
Do not guess, invent, or use outside knowledge.

Rules:
- If the answer is clearly present, give a concise, professional response.
- Use bullet points for multi-part answers.
- If the answer cannot be found in the document, respond exactly with: \
"This information is not available in the document."
- Keep your answer focused — do not pad with unnecessary explanation.

Document:
\"\"\"
{document_text[:14000]}
\"\"\"

Question: {question}

Answer:"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.3),
    )
    return response.text.strip()
