import os
import pandas as pd
import matplotlib.pyplot as plt

from evaluation.evaluate_system import run_evaluation_suite

# ======================================================
# Create results directory
# ======================================================

RESULTS_DIR = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ======================================================
# Models to evaluate
# ======================================================

models = [
    "tfidf",
    "bm25",
    "hybrid_parallel",
    "hybrid_serial"
]

all_results = []

print("\n========== STARTING EVALUATION ==========\n")

# ======================================================
# Run evaluation for each model
# ======================================================

for model in models:

    print(f"\nEvaluating {model}...\n")

    metrics = run_evaluation_suite(
        target_model=model
    )

    metrics["model"] = model

    all_results.append(metrics)

# ======================================================
# Save comparison table
# ======================================================

results_df = pd.DataFrame(all_results)

# جعل اسم النموذج أول عمود
cols = ["model"] + [
    c for c in results_df.columns
    if c != "model"
]

results_df = results_df[cols]

results_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "models_comparison.csv"
    ),
    index=False
)

print("\n===== FINAL COMPARISON =====")
print(results_df)

# ======================================================
# Draw charts
# ======================================================

metrics_to_plot = [
    "MAP@10",
    "Recall@10",
    "P@10",
    "nDCG@10",
    "MRR@10"
]

for metric in metrics_to_plot:

    if metric not in results_df.columns:
        continue

    plt.figure(figsize=(8, 5))

    plt.bar(
        results_df["model"],
        results_df[metric]
    )

    plt.title(f"{metric} Comparison")

    plt.xlabel("Model")

    plt.ylabel(metric)

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{metric}_comparison.png"
        )
    )

    plt.close()

print(
    "\nCharts saved successfully in evaluation/results/"
)