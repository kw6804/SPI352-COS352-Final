# Script for running Part 2 experiments across multiple models
# Asks each model every question in one batch
# (preserving history and context for the model)

# Print statements are for debugging/logging purposes
# and therefore can be removed for faster execution

# todo: Add concurrency so that prompts can be asked in parallel

import pandas as pd
import sys
import os
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from datetime import datetime
from src.config import MODELS, N_RUNS, SYSTEM_PROMPT
from src.functions import MODEL_FUNCS

# Function to safely save progress in case of termination
def safe_save(rows, final=False):
    if not rows:
        return

    df = pd.DataFrame(rows)

    if final:
        out_path = "data/final/part2_results.csv"
    else:
        # Save with timestamp to avoid overwriting anything
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = f"data/raw/part2_results_partial_{ts}.csv"

    df.to_csv(out_path, index=False)
    print(f"Saved progress → {out_path}")

def run():
    # Load the prompts
    prompts = pd.read_csv("data/prompts/nyc_prompts.csv")

    all_rows = []

    try:
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
                    print(f"    Prompting question {qid}...")

                    # Add the user message to the running conversation
                    messages.append({"role": "user", "content": statement})

                    # Call the model (adjust signature if your client differs)
                    if model_name == "chatgpt":
                        answer = ask_fn(messages, model=model_id)
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
                    print("\nPrompting executed.")
    except KeyboardInterrupt:
        print("\nInterupted by user. Saving progress...")
        safe_save(all_rows, final=False)
        raise

    except Exception as e:
        print("\nERROR OCCURRED! Saving progress...")
        traceback.print_exc()
        safe_save(all_rows, final=False)
        raise

    # Final save
    safe_save(all_rows, final=True)
    print("Fully completed")


if __name__ == "__main__":
    run()
