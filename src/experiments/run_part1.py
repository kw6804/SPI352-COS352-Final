# Script for running Part 1 experiments across multiple models
# Asks each model every question individually
# (clearing chat history between)

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
    """saving partial progress in case of api errors"""
    if not rows:
        return

    df = pd.DataFrame(rows)

    if final:
        out_path = "data/final/part1_results.csv"
    else:
        # Save with timestamp to avoid overwriting anything
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = f"data/raw/part1_results_partial_{ts}.csv"

    df.to_csv(out_path, index=False)
    print(f"Saved progress → {out_path}")


def run():
    # Load the prompts
    prompts = pd.read_csv("data/prompts/nyc_prompts.csv")

    all_rows = []

    try:
        for model_name, _ in MODELS.items():
            print(f"=== Running Part 1 for model: {model_name} ===")

            for _, row in prompts.iterrows():
                qid = row["question_id"]
                statement = row["statement"]

                print(f"\nPrompting {qid} now...")
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
