# Client for facilitating interactions with Claude model

import os
import anthropic
from dotenv import load_dotenv

from src.config import SYSTEM_PROMPT

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def ask_claude(messages, model="claude-3-haiku-20240307"):
    # Anthropic uses a slightly different structure
    user_messages = [m for m in messages if m["role"] == "user"]
    content = user_messages[-1]["content"]

    # Prompt using system prompt constraint
    response = client.messages.create(
        model=model,
        max_tokens=10,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": content}]
    )

    answer = response.content[0].text.strip().lower()

    # Error handling in case model does not comply
    if answer not in {"agree", "disagee", "neutral"}:
        return f"Model didn't respond in the desired way. It returned {answer[:10]}"

    return answer
