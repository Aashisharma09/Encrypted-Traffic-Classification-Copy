
import pandas as pd
import matplotlib

# Avoid Tkinter errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path


# STEP 1: Load five-run summary
file_path = Path("results/five_run_summary.csv")

df = pd.read_csv(file_path)

print("Five-run summary loaded successfully.")

print(df.to_string(index=False))


# STEP 2: Create results folder
results_folder = Path("results")
results_folder.mkdir(parents=True, exist_ok=True)


# STEP 3: Create two comparison graphs
fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 6)
)


# Accuracy comparison
axes[0].bar(
    df["Model"],
    df["Accuracy_Mean"],
    yerr=df["Accuracy_Std"],
    capsize=7,
    edgecolor="black"
)

axes[0].set_title("Mean Accuracy Across Five Runs")

axes[0].set_ylabel("Mean Accuracy")

axes[0].set_ylim(0, 1)

axes[0].tick_params(
    axis="x",
    rotation=20
)

axes[0].grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)


# Macro F1 comparison
axes[1].bar(
    df["Model"],
    df["Macro_F1_Mean"],
    yerr=df["Macro_F1_Std"],
    capsize=7,
    edgecolor="black"
)

axes[1].set_title("Mean Macro F1 Across Five Runs")

axes[1].set_ylabel("Mean Macro F1")

axes[1].set_ylim(0, 1)

axes[1].tick_params(
    axis="x",
    rotation=20
)

axes[1].grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)


# STEP 4: Overall graph title
fig.suptitle(
    "Five-Run Experimental Model Comparison",
    fontsize=15
)

plt.tight_layout()


# STEP 5: Save graph
graph_path = (
    results_folder / "five_run_model_comparison.png"
)

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# STEP 6: Completion message
print("\nGraph saved successfully:")
print(graph_path)

print("\n========== GRAPH GENERATION COMPLETED ==========")
