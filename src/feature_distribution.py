
import pandas as pd
import numpy as np
import matplotlib

# Avoid Tkinter errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path


# STEP 1: Load the dataset
data_path = Path("dataset/processed/cleaned_darknet.csv")

df = pd.read_csv(data_path)

print("Dataset loaded:", df.shape)


# STEP 2: Select important network traffic features
selected_features = [
    "Flow Duration",
    "Packet Length Mean",
    "Packet Length Std",
    "Flow IAT Mean",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean"
]


# STEP 3: Check whether features exist
available_features = [
    feature for feature in selected_features
    if feature in df.columns
]

missing_features = [
    feature for feature in selected_features
    if feature not in df.columns
]

print("\nSelected features:", available_features)

if missing_features:
    print("Features not found:", missing_features)

if not available_features:
    raise ValueError("None of the selected features were found.")


# STEP 4: Create results folder
results_folder = Path("results")
results_folder.mkdir(parents=True, exist_ok=True)


# STEP 5: Create distribution graphs
fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(16, 10)
)

axes = axes.flatten()


for i, feature in enumerate(available_features):

    # Convert values to numerical format
    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    # Remove missing and infinite values
    values = values.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    # Plot histogram
    axes[i].hist(
        values,
        bins=40,
        edgecolor="black"
    )

    axes[i].set_title(feature, fontsize=11)

    axes[i].set_xlabel("Feature Value")

    axes[i].set_ylabel("Frequency")

    # Logarithmic vertical scale helps show
    # both large and small frequencies
    axes[i].set_yscale("log")


# Hide unused graph areas
for j in range(len(available_features), len(axes)):
    axes[j].set_visible(False)


# STEP 6: Add overall title
fig.suptitle(
    "Distribution of Selected Network Traffic Features",
    fontsize=16
)

plt.tight_layout()


# STEP 7: Save graph
graph_path = results_folder / "feature_distributions.png"

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# STEP 8: Display descriptive statistics
print("\n========== FEATURE STATISTICS ==========")

print(
    df[available_features]
    .describe()
    .round(2)
)


# STEP 9: Completion message
print("\nGraph saved at:", graph_path)

print("\n========== FEATURE DISTRIBUTION ANALYSIS COMPLETED ==========")
