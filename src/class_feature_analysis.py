
import pandas as pd
import numpy as np
import matplotlib

# Avoid Tkinter errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

# STEP 1: Load dataset
data_path = Path("dataset/processed/cleaned_darknet.csv")
df = pd.read_csv(data_path)

print("Dataset loaded:", df.shape)

# STEP 2: Select features for comparison
selected_features = [
    "Flow Duration",
    "Packet Length Mean",
    "Flow IAT Mean"
]

target = "Label.1"

# STEP 3: Check required columns
for feature in selected_features:
    if feature not in df.columns:
        raise ValueError(f"Feature not found: {feature}")

# STEP 4: Prepare results folder
results_folder = Path("results")
results_folder.mkdir(parents=True, exist_ok=True)

# STEP 5: Prepare data for analysis
for feature in selected_features:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    df[feature] = df[feature].replace(
        [np.inf, -np.inf],
        np.nan
    )

# Remove rows with missing selected values
df = df.dropna(
    subset=selected_features + [target]
)

# Check values before logarithmic transformation
if (df[selected_features] < 0).any().any():
    raise ValueError(
        "Negative values found. Inspect the data before log transformation."
    )

# STEP 6: Create boxplots
classes = sorted(df[target].unique())

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(19, 7)
)

for i, feature in enumerate(selected_features):

    plot_data = []

    # Collect values for each application category
    for category in classes:

        values = df.loc[
            df[target] == category,
            feature
        ]

        # Sample up to 2000 records per category
        values = values.sample(
            n=min(2000, len(values)),
            random_state=42
        )

        # Apply log transformation
        values = np.log1p(values)

        plot_data.append(values)

    # Draw boxplot
    axes[i].boxplot(
        plot_data,
        tick_labels=classes,
        showfliers=False
    )

    axes[i].set_title(feature)

    axes[i].set_xlabel("Application Category")

    axes[i].set_ylabel("Log Transformed Value")

    axes[i].tick_params(
        axis="x",
        rotation=45
    )

# STEP 7: Add overall title
fig.suptitle(
    "Comparison of Network Traffic Features Across Application Categories",
    fontsize=15
)

plt.tight_layout()

# STEP 8: Save graph
graph_path = results_folder / "class_feature_comparison.png"

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# STEP 9: Print category-wise statistics
print("\n========== CATEGORY-WISE FEATURE MEDIANS ==========")

for feature in selected_features:

    print("\nFeature:", feature)

    print(
        df.groupby(target)[feature]
        .median()
        .round(2)
    )

# STEP 10: Completion message
print("\nGraph saved at:", graph_path)

print("\n========== CLASS FEATURE ANALYSIS COMPLETED ==========")