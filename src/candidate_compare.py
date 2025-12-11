# Compares the model responses to the responses of NYC mayoral candidates
# on the same questions to truly compare their political alignments

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load CSV files
STATEMENTS_FILE = "data/prompts/nyc_prompts.csv"  
MODEL_RESPONSES_FILE = "data/final/part1_results.csv"      
CANDIDATE_RESPONSES_FILE = "data/prompts/candidate_responses.csv" 

output_path = "data/final/alignment.png"

statements = pd.read_csv(STATEMENTS_FILE)
o_model_responses = pd.read_csv(MODEL_RESPONSES_FILE)
o_candidates = pd.read_csv(CANDIDATE_RESPONSES_FILE)

candidates = o_candidates.copy()
model_responses = o_model_responses.copy()

# Process files for comparison (keep overlapping columns, normalize)
model_responses = model_responses[["model", "question_id", "response"]]

model_responses["response"] = model_responses["response"].str.lower().str.strip()
for cand in ["Mamdani", "Cuomo", "Sliwa"]:
    candidates[cand] = candidates[cand].str.lower().str.strip()

merged = model_responses.merge(candidates, on="question_id", how="inner")

# Compute alignment for models and candidates
results = []

models = merged["model"].unique()
candidates_list = ["Mamdani", "Cuomo", "Sliwa"]

for model in models:
    subset = merged[merged["model"] == model]
    
    for cand in candidates_list:
        total = len(subset)
        matches = (subset["response"] == subset[cand]).sum()
        alignment = 100 * matches / total if total > 0 else 0
        
        results.append({
            "model": model,
            "candidate": cand,
            "alignment": alignment
        })

df_align = pd.DataFrame(results)

# Plot alignment
plt.figure(figsize=(12, 6))

colors = ['#4C72B0', '#DD8452', '#55A868']

x = np.arange(len(models))
width = 0.25

for i, cand in enumerate(candidates_list):
    cand_values = df_align[df_align["candidate"] == cand]["alignment"]
    plt.bar(x + (i - 1) * width, cand_values, width, label=cand, color=colors[i])

plt.xticks(x, models, rotation=20, fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.ylabel("Alignment (%)", fontsize=14, color='black')
plt.xlabel("LLM Model", fontsize=14, color='black')
plt.ylim(0, 100)
plt.title("Model Alignment with NYC Mayoral Candidates", fontsize=16, color='black')

legend = plt.legend(title="Candidate", title_fontsize=12, fontsize=11)
plt.setp(legend.get_title(), color='black')
plt.setp(legend.get_texts(), color='black')

ax = plt.gca()
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='lightgray')
ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(output_path, dpi=300)
plt.close()

# Now start process of calculating alignment by category
OUTPUT_DIR = "data/final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process files for comparison (keep overlapping columns, normalize)
o_model_responses["response"] = o_model_responses["response"].str.lower().str.strip()
for cand in ["Mamdani", "Cuomo", "Sliwa"]:
    o_candidates[cand] = o_candidates[cand].str.lower().str.strip()

o_model_responses = o_model_responses[["model", "question_id", "response"]]

merged = o_model_responses.merge(o_candidates, on="question_id", how="inner")

# Merge category info
merged = merged.merge(statements[["question_id", "category"]], on="question_id", how="left")

models = merged["model"].unique()
categories = merged["category"].dropna().unique()

# Create alignment graphs blocked by category
for cat in categories:
    subset_cat = merged[merged["category"] == cat]

    results = []

    for model in models:
        subset_model = subset_cat[subset_cat["model"] == model]

        for cand in candidates_list:
            total = len(subset_model)
            matches = (subset_model["response"] == subset_model[cand]).sum()
            alignment = 100 * matches / total if total > 0 else 0

            results.append({
                "model": model,
                "candidate": cand,
                "alignment": alignment
            })

    df_align = pd.DataFrame(results)

    # plot graphs
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.25

    for i, cand in enumerate(candidates_list):
        cand_values = df_align[df_align["candidate"] == cand]["alignment"]
        plt.bar(x + (i - 1) * width, cand_values, width, label=cand, color=colors[i])

    plt.xticks(x, models, rotation=20, fontsize=12)
    plt.ylabel("Alignment (%)", fontsize=14)
    plt.xlabel("LLM Model", fontsize=14)
    plt.ylim(0, 100)
    plt.title(f"Model Alignment with NYC Mayoral Candidates — {cat}", fontsize=16)
    plt.legend(title="Candidate")
    plt.tight_layout()

    # Save graph
    filename = f"{OUTPUT_DIR}/{cat.replace(' ', '_').replace('/', '_')}_alignment.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")
