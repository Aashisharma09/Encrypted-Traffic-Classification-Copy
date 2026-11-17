
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# STEP 1: Load dataset
df = pd.read_csv(
    "dataset/processed/cleaned_darknet.csv"
)

print("Dataset loaded:", df.shape)


# STEP 2: Separate features and target
X = df.drop(columns=["Label.1"])
y = df["Label.1"]

X = X.apply(pd.to_numeric, errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)

X = X.dropna(axis=1, how="all")


# STEP 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# STEP 4: Create Decision Tree model
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

predictions = model.predict(X_test)

print("Model trained successfully.")


# STEP 6: Calculate confusion matrix
labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)


# STEP 7: Plot confusion matrix
plt.figure(figsize=(12, 9))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(
    cmap="Blues",
    xticks_rotation=45,
    values_format="d",
    ax=plt.gca(),
    colorbar=False
)

plt.title(
    "Confusion Matrix - Decision Tree Classifier"
)

plt.tight_layout()


# STEP 8: Save confusion matrix graph
results_folder = Path("results")

results_folder.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    results_folder / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# STEP 9: Generate classification report
report = classification_report(
    y_test,
    predictions,
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========")

print(report)

with open(
    results_folder / "decision_tree_classification_report.txt",
    "w"
) as file:

    file.write(report)


# STEP 10: Save confusion matrix values
cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_df.to_csv(
    results_folder / "confusion_matrix.csv"
)


print("\nConfusion matrix saved successfully.")

print("Graph: results/confusion_matrix.png")

print("CSV: results/confusion_matrix.csv")

print("\n========== ANALYSIS COMPLETED ==========")
