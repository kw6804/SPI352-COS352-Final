import pandas as pd
from datetime import datetime

from src.config import MODELS, N_RUNS, SYSTEM_PROMPT

from src.models.chatgpt_client import ask_chatgpt
from src.models.claude_client import ask_claude
from src.models.gemini_client import ask_gemini


# Map model name → client function
MODEL_FUNCS = {
    "chatgpt": ask_chatgpt,
    "claude": ask_claude,
    "gemini": ask_gemini,
}


def run():
    # Load the prompts (assumes you’re using this file path)
    prompts = pd.read_csv("data/prompts/nyc_prompts.csv")

    all_rows = []

    for model_name, model_cfg in MODELS.items():
        print(f"=== Running Part 2 for model: {model_name} ===")

        # Start ONE conversation per model for all passes
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        ask_fn = MODEL_FUNCS[model_name]
        model_id = model_cfg["model"]

        # Pass = one full run through all questions
        for pass_idx in range(1, N_RUNS + 1):
            print(f"  Pass {pass_idx}/{N_RUNS}")

            # Loop through all questions in order
            for _, row in prompts.iterrows():
                qid = row["question_id"]
                statement = row["statement"]

                # Add the user message to the running conversation
                messages.append({"role": "user", "content": statement})

                # Call the model (adjust signature if your client differs)
                if model_name == "chatgpt":
                    answer = ask_fn(messages, model=model_id, temperature=0.0)
                elif model_name == "claude":
                    answer = ask_fn(messages, model=model_id)
                elif model_name == "gemini":
                    answer = ask_fn(messages, model=model_id)
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                # Record a row for this response
                all_rows.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "model": model_name,
                    "model_id": model_id,
                    "part": "part2_context",
                    "pass_idx": pass_idx,          # which of the 10 passes
                    "question_id": qid,
                    "statement": statement,
                    "response": answer,
                })

                # Add assistant reply back into context so it accumulates
                messages.append({"role": "assistant", "content": answer})

    # Save all results to CSV
    df = pd.DataFrame(all_rows)
    output_path = "data/raw/part2_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved Part 2 results to: {output_path}")


if __name__ == "__main__":
    run()
