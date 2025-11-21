import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def ask_gemini(messages, model="gemini-1.5-flash"):
    # Gemini takes only the latest user message
    latest_user_msg = [m for m in messages if m["role"] == "user"][-1]["content"]

    response = genai.GenerativeModel(model).generate_content(latest_user_msg)
    return response.text.strip()
