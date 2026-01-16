import google.generativeai as genai
import os
import base64
import re

from dotenv import load_dotenv
load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

import json
import re


def extract_json_from_text(text: str) -> dict:
    """
    Extracts and parses JSON from Gemini response safely.
    Handles markdown code fences if present.
    """
    # Remove markdown code fences
    text = re.sub(r"```json|```", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("Gemini returned invalid JSON")



def get_packaging_advice(image_bytes: bytes, fefco_standards: list):
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = f"""
You are a packaging expert.

Tasks:
1. Identify the product in the image
2. Estimate fragility (Low / Medium / High)
3. Recommend the best FEFCO box from this list ONLY:
{fefco_standards}

Decision criteria:
- Balance safety and cost
- Avoid over-packaging
- Consider transport durability

IMPORTANT RULES:
- Return ONLY valid JSON
- Do NOT use markdown
- Do NOT wrap output in ``` or ```json
- Do NOT add any extra text
- Keep the reasoning short and concise. Use the least amount of words possible.

Return output strictly in JSON format, no markdown code fences:
{{
  "product_type": "string",
  "fragility_level": "string",
  "recommended_fefco": "string",
  "reasoning": "string"
}}
"""

    response = model.generate_content(
        [
            prompt,
            {
                "mime_type": "image/jpeg",
                "data": image_bytes
            }
        ]
    )

    raw_text = response.text
    parsed_json = extract_json_from_text(raw_text)

    return parsed_json
