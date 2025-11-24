# Client for facilitating interactions with ChatGPT model

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def ask_chatgpt(messages, model="gpt-5-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    answer = response.choices[0].message.content.strip().lower()
    
    # Error handling in case model does not comply
    if answer not in {"agree", "disagree", "neutral"}:
        return f"Model didn't respond in the desired way. It returned {answer[:10]}"

    return answer
