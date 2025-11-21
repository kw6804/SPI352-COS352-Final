import os
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def ask_claude(messages, model="claude-3-haiku-20240307"):
    # Anthropic uses a slightly different structure
    user_messages = [m for m in messages if m["role"] == "user"]

    content = user_messages[-1]["content"]

    response = client.messages.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": content}]
    )
    return response.content[0].text.strip()
