
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path


# STEP 1: Load five-run class summary
file_path = Path("results/five_run_class_summary.csv")

df = pd.read_csv(file_path)

print("Class-wise results loaded successfully.")

print("\nApplication-wise performance:")
print(df.round(4).to_string(index=False))


# STEP 2: Create results folder
results_folder = Path("results")
results_folder.mkdir(parents=True, exist_ok=True)


# STEP 3: Create F1 score comparison table
pivot_df = df.pivot(
    index="Application",
    columns="Model",
    values="F1_Mean"
)

print("\n========== APPLICATION-WISE F1 COMPARISON ==========")

print(pivot_df.round(4).to_string())


# STEP 4: Create grouped bar graph
ax = pivot_df.plot(
    kind="bar",
    figsize=(14, 7),
    width=0.8,
    edgecolor="black"
)

plt.title(
    "Application-wise F1 Score Comparison Across Models"
)

plt.xlabel("Application Category")

plt.ylabel("Mean F1 Score")

plt.ylim(0, 1)

plt.xticks(rotation=35, ha="right")

plt.legend(
    title="Machine Learning Models"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()


# STEP 5: Save graph
plt.savefig(
    results_folder / "class_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# STEP 6: Save comparison table
pivot_df.to_csv(
    results_folder / "class_performance_comparison.csv"
)


print("\nGraph saved:")
print("results/class_performance_comparison.png")

print("\nTable saved:")
print("results/class_performance_comparison.csv")

print("\n========== ANALYSIS COMPLETED ==========")
