# Script for running Part 1 experiments across multiple models
# Asks each model every question individually
# (clearing chat history between)

# Print statements are for debugging/logging purposes
# and therefore can be removed for faster execution
# Progress can bar can similarly be removed

import pandas as pd
import sys
import os
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from datetime import datetime
from src.config import MODELS, N_RUNS, SYSTEM_PROMPT
from src.functions import MODEL_FUNCS

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm 

# Function to safely save progress in case of termination
def safe_save(rows, final=False):
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

# Create and serve prompt for one model
def handle_prompt(model_name, qid, statement, i):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": statement})

    try:
        answer = MODEL_FUNCS[model_name](messages)
    except Exception as e:
        answer = f"ERROR: {str(e)}"
    
    return {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model_name,
            "part": "part1",
            "run_idx": i,
            "question_id": qid,
            "statement": statement,
            "response": answer,
        }

# Handle a single experiment run (thread)
def handle_run(model_name, prompts, i):
    results = []

    for _, row in prompts.iterrows():
        qid = row["question_id"]
        statement = row["statement"]

        result = handle_prompt(model_name, qid, statement, i)
        results.append(result)

    return results

# Concurrently run part1 experiment
def run():
    # Load the prompts
    prompts = pd.read_csv("data/prompts/nyc_prompts.csv")

    all_rows = []

    total_runs = N_RUNS * len(MODELS) * len(prompts)
    progress = tqdm(total=total_runs, desc="Running experiment...")

    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for model_name in MODELS:
                for i in range(1, N_RUNS+1):
                    futures.append(
                        executor.submit(
                            handle_run,
                            model_name,
                            prompts,
                            i
                        )
                    )
                        
            for future in as_completed(futures):
                run_rows = future.result()

                for row in run_rows:
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
