import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

COLOR_MAP = {
    "agree": 1,
    "neutral": 0,
    "disagree": -1
}

def truncate(text, max_len=40):
    return text if len(text) <= max_len else text[:max_len] + "…"

def handle_heatmap(csv_path, output_path):
    df = pd.read_csv(csv_path)

    df["response"] = df["response"].astype(str).str.strip().str.lower()
    df["score"] = df["response"].map(COLOR_MAP)

    # Average score for conflicted responses
    grouped = (
        df.groupby(["model", "statement"])["score"]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot(index="model", columns="statement", values="score")
    
    pivot = pivot.sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    # Create and plot colormap
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="lightgrey") 
    norm = plt.Normalize(vmin=-1, vmax=1)

    plt.figure(figsize=(25, 8))
    plt.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)
    
    plt.yticks(np.arange(len(pivot.index)), pivot.index, fontsize=12)
    plt.xticks(
        np.arange(len(pivot.columns)),
        [truncate(s, 40) for s in pivot.columns],
        fontsize=6,
        rotation=70,
        ha="right"
    )

    # Add colorbar along axis
    cbar = plt.colorbar()
    cbar.set_label("Average Score  (-1=Disagree, 0=Neutral, 1=Agree)", fontsize=12)

    plt.gcf().subplots_adjust(right=0.5)

    plt.gcf().subplots_adjust(bottom=0.35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved heatmap → {output_path}")


def main():
    handle_heatmap(
        "data/final/part1_results.csv",
        "data/final/heatmap_part1.png",
    )

    handle_heatmap(
        "data/final/part2_results.csv",
        "data/final/heatmap_part2.png",
    )

if __name__ == "__main__":
    main()
