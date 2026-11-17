
import pandas as pd
import numpy as np

df = pd.read_csv(
    "dataset/processed/cleaned_darknet.csv"
)

X = df.drop(columns=["Label.1"])

X = X.apply(pd.to_numeric, errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)

print("Original dataset shape:", df.shape)

print("Original feature count:", X.shape[1])

print("Completely empty columns:", X.isna().all().sum())

# Remove completely empty columns
X = X.dropna(axis=1, how="all")

# Identify constant columns
constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=True) <= 1
]

print("Constant columns:", len(constant_columns))

# Remove constant columns
X = X.drop(columns=constant_columns)

print("Final usable feature count:", X.shape[1])

print("\nFinal feature names:")
print(X.columns.tolist())

print("\n========== AUDIT COMPLETED ==========")
