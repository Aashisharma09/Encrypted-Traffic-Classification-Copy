
import pandas as pd
import numpy as np
import matplotlib

# Prevent Tkinter display errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


# STEP 1: Load dataset
data_path = Path("dataset/processed/cleaned_darknet.csv")

df = pd.read_csv(data_path)

print("Dataset loaded:", df.shape)


# STEP 2: Separate features and target
X = df.drop(columns=["Label.1"])
y = df["Label.1"]

# Convert features to numeric values
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Remove completely empty columns
X = X.dropna(axis=1, how="all")


# STEP 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# STEP 4: Create Decision Tree pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])


# STEP 5: Train model
print("\nTraining Decision Tree...")

model.fit(X_train, y_train)

print("Model trained successfully.")


# STEP 6: Extract feature importance
classifier = model.named_steps["classifier"]

importance = classifier.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})


# STEP 7: Sort features by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP 20 IMPORTANT FEATURES ==========")

print(
    importance_df.head(20).to_string(index=False)
)


# STEP 8: Save feature importance
results_folder = Path("results")
results_folder.mkdir(parents=True, exist_ok=True)

importance_df.to_csv(
    results_folder / "feature_importance.csv",
    index=False
)


# STEP 9: Select top 20 features
top_features = importance_df.head(20)

# Reverse order for horizontal graph
top_features = top_features.iloc[::-1]


# STEP 10: Create graph
plt.figure(figsize=(12, 8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"],
    edgecolor="black"
)

plt.xlabel("Feature Importance Score")

plt.ylabel("Network Traffic Features")

plt.title(
    "Top 20 Important Features for Encrypted Traffic Classification"
)

plt.grid(
    axis="x",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()


# STEP 11: Save graph
plt.savefig(
    results_folder / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nFeature importance saved successfully.")

print("CSV: results/feature_importance.csv")

print("Graph: results/feature_importance.png")

print("\n========== ANALYSIS COMPLETED ==========")
