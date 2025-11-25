# Script for running Part 2 experiments across multiple models
# Asks each model every question in one batch
# (preserving history and context for the model)

# Print statements are for debugging/logging purposes
# and therefore can be removed for faster execution
# Progress bar can similarly be removed

import pandas as pd
import sys
import os
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from datetime import datetime
from src.config import MODELS, N_RUNS, SYSTEM_PROMPT
from src.functions import MODEL_FUNCS

from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from tqdm import tqdm

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

# Create and serve prompt for one model
def handle_prompt(model_name, qid, statement, i, messages):
    try:
        # Adds context from previous messages
        messages.append({"role": "user", "content": statement})

        answer = MODEL_FUNCS[model_name](messages)
        messages.append({"role": "assistant", "content": answer})

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model_name,
            "part": "part2_context",
            "run_idx": i,
            "question_id": qid,
            "statement": statement,
            "response": answer,
        }

    except Exception as e:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model_name,
            "part": "part2_context",
            "run_idx": i,
            "question_id": qid,
            "statement": statement,
            "response": f"ERROR: {str(e)}",
        }

# Keeps a copy of messages for each thread
def handle_conversation(model_name, model_cfg, prompts, i):
    rows = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for _, row in prompts.iterrows():
        qid = row["question_id"]
        statement = row["statement"]

        result = handle_prompt(
            model_name,
            qid,
            statement,
            i,
            messages
        )

        rows.append(result)

    return rows


# Concurrently run part2 experiment
def run():
    # Load the prompts
    prompts = pd.read_csv("data/prompts/nyc_prompts.csv")

    all_rows = []

    total_runs = N_RUNS * len(MODELS)
    progress = tqdm(total=total_runs, desc="Running experiment...")

    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []

            for model_name, model_cfg in MODELS.items():
                print(f"=== Running Part 2 for model: {model_name} ===")

                # Pass = one full run through all questions
                for i in range(1, N_RUNS + 1):
                    futures.append(
                        executor.submit(
                            handle_conversation,
                            model_name,
                            model_cfg,
                            prompts,
                            i
                        )
                    )

            for future in as_completed(futures):
                for row in future.result():
                    all_rows.append(row)
                progress.update(1)
                                 
    except KeyboardInterrupt:
        print("\nInterupted by user. Saving progress...")
        safe_save(all_rows, final=False)
        progress.close()
        raise

    except Exception as e:
        print("\nERROR OCCURRED! Saving progress...")
        traceback.print_exc()
        safe_save(all_rows, final=False)
        progress.close()
        raise

    # Final save
    safe_save(all_rows, final=True)
    print("Fully completed")
    progress.close()


if __name__ == "__main__":
    run()
