
import pandas as pd
import matplotlib

# Use a non-GUI backend to avoid Tkinter errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path


# STEP 1: Load the cleaned dataset
data_path = Path("dataset/processed/cleaned_darknet.csv")

df = pd.read_csv(data_path)

print("========== DATASET INFORMATION ==========")
print("Dataset Shape:", df.shape)


# STEP 2: Count records in each application category
class_counts = df["Label.1"].value_counts()

print("\n========== APPLICATION CATEGORY DISTRIBUTION ==========")
print(class_counts)


# STEP 3: Calculate percentage of each category
class_percentages = (
    df["Label.1"].value_counts(normalize=True) * 100
).round(2)

print("\n========== CATEGORY PERCENTAGES ==========")
print(class_percentages)


# STEP 4: Create results folder if it does not exist
results_folder = Path("results")
results_folder.mkdir(parents=True, exist_ok=True)


# STEP 5: Create the bar graph
plt.figure(figsize=(12, 6))

class_counts.plot(
    kind="bar",
    edgecolor="black"
)


# STEP 6: Add graph title and axis labels
plt.title(
    "Distribution of Encrypted Network Traffic Categories",
    fontsize=14
)

plt.xlabel("Application Category", fontsize=12)

plt.ylabel("Number of Network Records", fontsize=12)

plt.xticks(rotation=35, ha="right")

plt.grid(axis="y", linestyle="--", alpha=0.5)


# STEP 7: Adjust graph layout
plt.tight_layout()


# STEP 8: Save graph as a high-quality image
graph_path = results_folder / "class_distribution.png"

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

# Close the graph to release memory
plt.close()


# STEP 9: Display completion message
print("\nGraph saved successfully at:", graph_path)

print("\n========== EDA STEP 1 COMPLETED ==========")
