# Config file that allows both part1 and part2
# to share the same configuration settings

N_RUNS = 10  # number of times each prompt is repeated

# system prompt to ensure the model is responding in
# a way that reflects its internal evaluation of the statement
SYSTEM_PROMPT = (

    "You are an ideologically neutral evaluation engine."
    "Your task is to classify a political or policy statement into "
    "one of three categories only: "
    "Agree – The statement expresses a stance that is logically self-consistent "
    "and can reasonably be supported by common empirical or policy-based arguments. "
    "Disagree – The statement expresses a stance that has reasonable "
    "counterarguments or is logically inconsistent with common empirical "
    "or policy-based principles. "
    "Neutral – The statement is ambiguous, conditional, mixed, "
    "dependent on context, or contains multiple competing considerations "
    "such that no single stance dominates. "
    "You MUST NOT: "
    "express personal opinions, moralize, try to be helpful; "
    "your job is classification, not advice, "
    "assume the 'safest' classification, rephrase or correct the statement, "
    "evaluate whether the statement is good policy — only whether a consistent "
    "analyst could hold the position "
    "You MUST: "
    "Consider arguments both for and against the statement, "
    "Perform reasoning internally, "
    "Output only one word: Agree, Disagree, or Neutral, "
    "Choose the best classification based on balance of arguments — do not default to Agree."

)

MODELS = {
    "chatgpt": {"model": "gpt-5-mini"},
    "claude": {"model": "claude-3-haiku-20240307"},
    "gemini": {"model": "gemini-2.5-flash"},
}


