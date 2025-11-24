# Allows both part1 and part2 to share model function mappings

from src.models.chatgpt_client import ask_chatgpt
from src.models.claude_client import ask_claude
from src.models.gemini_client import ask_gemini

MODEL_FUNCS = {
    "chatgpt": ask_chatgpt,
    "claude": ask_claude,
    "gemini": ask_gemini,
}