
import pandas as pd
import numpy as np
import matplotlib

# Avoid Tkinter display errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path


# STEP 1: Load dataset
data_path = Path("dataset/processed/cleaned_darknet.csv")
df = pd.read_csv(data_path)

print("Dataset loaded:", df.shape)


# STEP 2: Select numerical features
X = df.drop(columns=["Label.1"])

X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Count columns with all missing values
all_missing = X.columns[X.isna().all()].tolist()

# Remove columns with all missing values
X = X.dropna(axis=1, how="all")

# Identify constant columns
constant_columns = X.columns[
    X.nunique(dropna=True) <= 1
].tolist()

# Remove constant columns
X = X.drop(columns=constant_columns)

print("\n========== FEATURE INFORMATION ==========")
print("Original numerical columns:", 79)
print("Columns entirely missing:", len(all_missing))
print("Constant columns:", len(constant_columns))
print("Usable numerical features:", X.shape[1])


# STEP 3: Calculate correlation matrix
correlation_matrix = X.corr(method="pearson")

print("\nCorrelation matrix calculated successfully.")


# STEP 4: Calculate average absolute correlation
# Create a separate writable NumPy array
absolute_corr = correlation_matrix.abs().to_numpy(
    copy=True
)

# Ignore self-correlation along the diagonal
np.fill_diagonal(absolute_corr, np.nan)

# Calculate average absolute correlation
average_correlation = pd.Series(
    np.nanmean(absolute_corr, axis=1),
    index=correlation_matrix.columns
)

# Select 20 features with highest average correlation
top_features = (
    average_correlation
    .sort_values(ascending=False)
    .head(20)
    .index
)

# Create smaller correlation matrix
top_corr = correlation_matrix.loc[
    top_features, top_features
]


# STEP 5: Create results folder
results_folder = Path("results")
results_folder.mkdir(parents=True, exist_ok=True)


# STEP 6: Create heatmap
plt.figure(figsize=(14, 11))

heatmap = plt.imshow(
    top_corr,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(
    heatmap,
    label="Correlation Coefficient"
)

plt.xticks(
    range(len(top_features)),
    top_features,
    rotation=90,
    fontsize=8
)

plt.yticks(
    range(len(top_features)),
    top_features,
    fontsize=8
)

plt.title(
    "Correlation Heatmap of Selected Network Traffic Features",
    fontsize=14
)

plt.tight_layout()


# STEP 7: Save heatmap
plt.savefig(
    results_folder / "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# STEP 8: Find strongest correlated feature pairs
upper_triangle = np.triu(
    np.ones(correlation_matrix.shape),
    k=1
).astype(bool)

correlation_pairs = (
    correlation_matrix.where(upper_triangle)
    .stack()
    .sort_values(key=abs, ascending=False)
)

print("\n========== TOP 10 CORRELATED FEATURE PAIRS ==========")

print(correlation_pairs.head(10))


# STEP 9: Completion message
print("\nHeatmap saved in results/correlation_heatmap.png")

print("\n========== CORRELATION ANALYSIS COMPLETED ==========")
