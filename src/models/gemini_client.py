# Client for facilitating interactions with Claude model

import os
import google.generativeai as genai
from dotenv import load_dotenv

from src.config import SYSTEM_PROMPT

# Gemini requires API key setup directly in client
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def ask_gemini(messages, model="gemini-2.5-flash"):
    # Gemini takes only the latest user message
    latest_user_msg = [m for m in messages if m["role"] == "user"][-1]["content"]

    # Prompt using system prompt constraint
    prompt = f"{SYSTEM_PROMPT}\n\nUser: {latest_user_msg}"
    response = genai.GenerativeModel(model).generate_content(prompt)
    answer = response.text.strip().lower()

    # Error handling in case model does not comply
    if answer not in {"agree", "disagree", "neutral"}:
        return f"Model didn't respond in the desired way. It returned {answer[:10]}"

    return answer
