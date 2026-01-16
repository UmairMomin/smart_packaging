import google.generativeai as genai
import os

from dotenv import load_dotenv
load_dotenv()

# Make sure your API key is loaded
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

models = genai.list_models()

print("AVAILABLE MODELS:\n")

for m in models:
    print(f"Name: {m.name}")
    print(f"  Supported methods: {m.supported_generation_methods}")
    print("-" * 50)
