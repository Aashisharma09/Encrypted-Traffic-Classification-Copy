
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# ==========================================
# 1. LOAD CLEANED DATASET
# ==========================================

data_path = Path("dataset/processed/cleaned_darknet.csv")
df = pd.read_csv(data_path)

print("Dataset loaded successfully:", df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["Label.1"])
y = df["Label.1"]

print("Original number of features:", X.shape[1])


# ==========================================
# 3. DATA CLEANING
# ==========================================

# Convert all feature columns into numeric values
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values with missing values
X = X.replace([np.inf, -np.inf], np.nan)

# Identify constant columns
constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=True) <= 1
]

# Remove constant columns
X = X.drop(columns=constant_columns)

print("Constant columns removed:", len(constant_columns))
print("Final number of usable features:", X.shape[1])

# Replace missing values with column median
X = X.fillna(X.median())

# ==========================================
# 4. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 5. DEFINE MACHINE LEARNING MODELS
# ==========================================

models = {

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),

        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])
}


# ==========================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================

results = []

for name, model in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # Calculate Macro F1 score
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1
    })

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))

    # Display classification report
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


# ==========================================
# 7. SAVE MODEL COMPARISON RESULTS
# ==========================================

results_folder = Path("results")
results_folder.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    results_folder / "model_comparison.csv",
    index=False
)

print("\n========== FINAL MODEL COMPARISON ==========")

print(results_df.to_string(index=False))

print("\nResults saved in results/model_comparison.csv")

print("\n========== MODEL TRAINING COMPLETED ==========")
