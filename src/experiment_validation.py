
import pandas as pd
import numpy as np
from pathlib import Path

# Use a non-interactive backend to avoid Tkinter errors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

data_path = Path("dataset/processed/cleaned_darknet.csv")

df = pd.read_csv(data_path)

X = df.drop(columns=["Label.1"])
y = df["Label.1"]

print("\n========== DATASET INFORMATION ==========")

print("Dataset shape:", df.shape)
print("Original features:", X.shape[1])


# ==========================================
# 2. DATA CLEANING
# ==========================================

X = X.apply(pd.to_numeric, errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)

# Remove constant features

constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=True) <= 1
]

X = X.drop(columns=constant_columns)

print("Constant columns removed:", len(constant_columns))
print("Final usable features:", X.shape[1])


# ==========================================
# 3. EXPERIMENT SETTINGS
# ==========================================

random_seeds = [42, 21, 7, 100, 2026]

all_results = []

results_folder = Path("results")

results_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 4. FIVE-RUN EXPERIMENT
# ==========================================

for run_number, seed in enumerate(random_seeds, start=1):

    print("\n====================================")
    print("EXPERIMENTAL RUN:", run_number)
    print("Random State:", seed)
    print("====================================")

    # Split dataset into training and testing

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )

    # Define models

    models = {

        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                random_state=seed,
                class_weight="balanced"
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced"
            ))
        ]),

        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ])
    }

    # Train and evaluate models

    for name, model in models.items():

        print("\nTraining:", name)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        all_results.append({

            "Run": run_number,
            "Random State": seed,
            "Model": name,
            "Accuracy": accuracy,
            "Macro F1": macro_f1

        })

        print("Accuracy:", round(accuracy, 4))
        print("Macro F1:", round(macro_f1, 4))


# ==========================================
# 5. SAVE DETAILED RESULTS
# ==========================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    results_folder / "five_run_detailed_results.csv",
    index=False
)


# ==========================================
# 6. CALCULATE SUMMARY STATISTICS
# ==========================================

summary = results_df.groupby("Model").agg(

    Accuracy_Mean=("Accuracy", "mean"),
    Accuracy_Std=("Accuracy", "std"),

    Macro_F1_Mean=("Macro F1", "mean"),
    Macro_F1_Std=("Macro F1", "std")

).reset_index()


summary = summary.sort_values(
    by="Accuracy_Mean",
    ascending=False
)


# Save summary

summary.to_csv(
    results_folder / "five_run_model_summary.csv",
    index=False
)


# ==========================================
# 7. DISPLAY FINAL RESULTS
# ==========================================

print("\n========== FIVE-RUN MODEL SUMMARY ==========")

print(summary.round(4).to_string(index=False))

print("\nDetailed results saved successfully.")

print("results/five_run_detailed_results.csv")

print("results/five_run_model_summary.csv")


# ==========================================
# 8. CREATE COMPARISON GRAPH
# ==========================================

plt.figure(figsize=(12, 6))

x = np.arange(len(summary))

width = 0.35

plt.bar(
    x - width / 2,
    summary["Accuracy_Mean"],
    width,
    yerr=summary["Accuracy_Std"],
    capsize=5,
    label="Mean Accuracy"
)

plt.bar(
    x + width / 2,
    summary["Macro_F1_Mean"],
    width,
    yerr=summary["Macro_F1_Std"],
    capsize=5,
    label="Mean Macro F1"
)

plt.xticks(
    x,
    summary["Model"],
    rotation=15
)

plt.ylim(0, 1.05)

plt.xlabel("Machine Learning Models")

plt.ylabel("Performance Score")

plt.title(
    "Five-Run Experimental Model Comparison"
)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

# Save graph without opening a window

plt.savefig(
    results_folder / "five_run_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nGraph saved successfully.")

print("results/five_run_model_comparison.png")

print("\n========== EXPERIMENT COMPLETED ==========")
