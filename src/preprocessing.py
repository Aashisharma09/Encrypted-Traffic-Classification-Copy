
import pandas as pd
import numpy as np
from pathlib import Path

# Project folders
input_file = Path("dataset/raw/Darknet.CSV")
output_folder = Path("dataset/processed")
output_file = output_folder / "cleaned_darknet.csv"

# Create the processed folder if needed
output_folder.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")

# Read the original dataset
df = pd.read_csv(input_file, low_memory=False)

print("Original rows:", len(df))

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

# Use Label.1 as our application classification target
df["Label.1"] = df["Label.1"].astype(str).str.strip().str.upper()

# Standardize application labels
df["Label.1"] = df["Label.1"].replace({
    "AUDIO-STREAMING": "AUDIO-STREAMING",
    "VIDEO-STREAMING": "VIDEO-STREAMING",
    "FILE-TRANSFER": "FILE-TRANSFER"
})

# Remove rows without a valid target label
df = df.dropna(subset=["Label.1"])
df = df[df["Label.1"] != "NAN"]

# Remove identifiers and the alternative Tor/Non-Tor label
columns_to_drop = [
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Label"
]

df = df.drop(columns=columns_to_drop, errors="ignore")

# Convert infinite values into missing values
df = df.replace([np.inf, -np.inf], np.nan)

# Remove rows with missing values
df = df.dropna()

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("\nCleaning completed!")
print("Cleaned rows:", len(df))
print("Number of columns:", len(df.columns))
print("\nApplication categories:")
print(df["Label.1"].value_counts())
print("\nSaved to:", output_file)
