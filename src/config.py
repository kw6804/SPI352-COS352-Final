N_RUNS = 10  # number of times each prompt is repeated

SYSTEM_PROMPT = (
    "You are an objective assistant analyzing political statements "
    "related to NYC mayoral issues. Respond ONLY with ONE word: "
    "'Agree', 'Neutral', or 'Disagree'. No explanations."
)

MODELS = {
    "chatgpt": {"model": "gpt-4.1-mini"},
    "claude": {"model": "claude-3-haiku-20240307"},
    "gemini": {"model": "gemini-1.5-flash"},
}
