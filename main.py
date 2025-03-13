import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Suppress UserWarnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Create output directories if they don’t exist
os.makedirs("output/plots", exist_ok=True)
os.makedirs("output/comparisons", exist_ok=True)

# --------------------------------------------------------
# 1. Load & Preprocess Data
# --------------------------------------------------------
col_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

df_train = pd.read_csv("data/adult.data", header=None, names=col_names)
df_test = pd.read_csv("data/adult.test", header=None, names=col_names, skiprows=1)

# Replace missing values with NaN
df_train.replace("?", np.nan, inplace=True)
df_test.replace("?", np.nan, inplace=True)

# Drop rows with missing values
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

# Reset index
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

# Clean up "income" in df_test (remove trailing ".")
df_test["income"] = df_test["income"].apply(lambda x: x.replace(".", "").strip())

# Convert "age" to numeric
df_train["age"] = pd.to_numeric(df_train["age"], errors="coerce").astype(int)
df_test["age"] = pd.to_numeric(df_test["age"], errors="coerce").astype(int)

# --------------------------------------------------------
# 2. Prepare Data for Training (Exclude `fnlwgt`)
# --------------------------------------------------------
X_train = df_train.drop(["income", "fnlwgt"], axis=1)
y_train = df_train["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

X_test = df_test.drop(["income", "fnlwgt"], axis=1)
y_test = df_test["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

# --------------------------------------------------------
# 3. Define Preprocessing & Model Pipelines
# --------------------------------------------------------
numeric_features = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
categorical_features = ["workclass", "education", "marital-status", "occupation", 
                        "relationship", "race", "sex", "native-country"]

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
rf_pipeline.fit(X_train, y_train)

# --------------------------------------------------------
# 4. Feature Correlation Heatmap (Including Income)
# --------------------------------------------------------
# Add "income" as a numeric feature
df_train["income"] = y_train  # Ensure income is in numerical format (0, 1)

# Compute correlation matrix including income
df_numeric = df_train[numeric_features + ["income"]]
correlation_matrix = df_numeric.corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix (Including Income)")
plt.tight_layout()
plt.savefig("output/plots/feature_correlation_income.png")
plt.close()

# --------------------------------------------------------
# 5. Predicted vs. Actual Cases
# --------------------------------------------------------
y_pred_rf = rf_pipeline.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Predicted vs. Actual Cases (Random Forest)")
plt.tight_layout()
plt.savefig("output/plots/predicted_vs_actual.png")
plt.close()

# --------------------------------------------------------
# 6. Fairness Interpretation & Save Outputs
# --------------------------------------------------------
def interpret_group_importance(group_col, model_pipeline, output_file):
    """Determines feature importance for predicting >50K and saves it in a text file."""
    with open(output_file, "w") as f:
        f.write(f"=== Importance of {group_col} in Predicting High Income ===\n\n")

        feature_importances = model_pipeline.named_steps["rf"].feature_importances_

        cat_features = model_pipeline.named_steps["preprocessor"].transformers_[1][2]
        one_hot_encoder = model_pipeline.named_steps["preprocessor"].transformers_[1][1]
        feature_names = list(numeric_features) + [
            f"{col}={val}" for col, values in zip(cat_features, one_hot_encoder.categories_) for val in values[1:]
        ]

        group_features = {name: importance for name, importance in zip(feature_names, feature_importances) if group_col in name}
        sorted_group_features = sorted(group_features.items(), key=lambda x: x[1], reverse=True)

        for name, importance in sorted_group_features:
            f.write(f"{name}: {importance:.4f}\n")

        if not sorted_group_features:
            f.write("No significant impact found for this category.\n")

# Generate feature importance for race, occupation, and native-country
for category in ["race", "occupation", "native-country"]:
    output_path = f"output/comparisons/{category}.txt"
    interpret_group_importance(category, rf_pipeline, output_path)

# --------------------------------------------------------
# 7. Income Analysis by Education (Formatted List)
# --------------------------------------------------------
def analyze_education_impact(output_file):
    """Compares education levels and their likelihood of earning >50K."""
    overall_high_income_rate = y_test.mean()
    education_results = []

    for edu_level in X_test["education"].unique():
        idx = (X_test["education"] == edu_level)
        if idx.sum() == 0:
            continue

        X_subset = X_test[idx]
        y_pred_subset = rf_pipeline.predict(X_subset)
        group_high_income_rate = np.mean(y_pred_subset)

        education_results.append((edu_level, group_high_income_rate))

    # Sort results in descending order of likelihood
    education_results.sort(key=lambda x: x[1], reverse=True)

    # Save results in formatted list
    with open(output_file, "w") as f:
        f.write("=== Income Analysis by Education ===\n\n")
        for edu_level, rate in education_results:
            f.write(f"{edu_level}: {rate:.4f}\n")

# Save education impact analysis
analyze_education_impact("output/comparisons/education.txt")


# --------------------------------------------------------
# 8. Train Bias-Reduced Model (Excluding Race, Gender, and Country)
# --------------------------------------------------------
X_train_fair = X_train.drop(["race", "sex", "native-country"], axis=1)
X_test_fair = X_test.drop(["race", "sex", "native-country"], axis=1)

rf_pipeline_fair = Pipeline(steps=[
    ("preprocessor", ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), [
            "workclass", "education", "marital-status", "occupation", "relationship"
        ])
    ])),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the bias-reduced model
rf_pipeline_fair.fit(X_train_fair, y_train)

# --------------------------------------------------------
# 9. Compare Model Performance
# --------------------------------------------------------
def evaluate_model(model, X_test_data, y_test_data):
    """Evaluates a model and returns performance metrics."""
    y_pred = model.predict(X_test_data)
    return {
        "Accuracy": accuracy_score(y_test_data, y_pred),
        "Precision": precision_score(y_test_data, y_pred),
        "Recall": recall_score(y_test_data, y_pred),
        "F1 Score": f1_score(y_test_data, y_pred),
    }

# Evaluate original and bias-reduced models
original_metrics = evaluate_model(rf_pipeline, X_test, y_test)
fair_metrics = evaluate_model(rf_pipeline_fair, X_test_fair, y_test)

# Save results to a file
comparison_path = "output/comparisons/model_comparison.txt"
with open(comparison_path, "w") as f:
    f.write("=== Model Performance Comparison ===\n\n")
    f.write("Original Model (With Race, Gender, and Country):\n")
    for key, value in original_metrics.items():
        f.write(f"  {key}: {value:.4f}\n")
    f.write("\nBias-Reduced Model (Without Race, Gender, and Country):\n")
    for key, value in fair_metrics.items():
        f.write(f"  {key}: {value:.4f}\n")

print(" All outputs saved in 'output/' directory.")
