import pandas as pd
from datetime import datetime
from src.config import MODELS, N_RUNS, SYSTEM_PROMPT

from src.models.chatgpt_client import ask_chatgpt
from src.models.claude_client import ask_claude
from src.models.gemini_client import ask_gemini


MODEL_FUNCS = {
    "chatgpt": ask_chatgpt,
    "claude": ask_claude,
    "gemini": ask_gemini,
}

def run():
    prompts = pd.read_csv("data/prompts/nyc_prompts.csv")

    all_rows = []

    for model_name in MODEL_FUNCS:
        for _, row in prompts.iterrows():
            qid = row["question_id"]
            statement = row["statement"]

            # Start fresh conversation
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            for i in range(N_RUNS):
                messages.append({"role": "user", "content": statement})

                answer = MODEL_FUNCS[model_name](messages)

                all_rows.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "model": model_name,
                    "part": "part1",
                    "run_idx": i+1,
                    "question_id": qid,
                    "statement": statement,
                    "response": answer,
                })

                # Add assistant response back to context
                messages.append({"role": "assistant", "content": answer})

    df = pd.DataFrame(all_rows)
    df.to_csv("data/raw/part1_results.csv", index=False)
    print("Saved: data/raw/part1_results.csv")

if __name__ == "__main__":
    run()
