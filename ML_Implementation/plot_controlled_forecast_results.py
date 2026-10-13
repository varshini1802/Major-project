"""Create paper-ready figures directly from saved corrected-v4 result tables."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence"
ABL = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_feature_ablation"
OUT = BASE / "figures"


def main():
    OUT.mkdir(exist_ok=True)
    result = pd.read_csv(BASE / "test_results.csv")
    selected = result[result.model.isin(["GRU", "LSTM", "LogisticRegression", "Majority baseline"])]
    metrics = [("test_macro_f1", "Macro F1"), ("test_weighted_f1", "Weighted F1"),
               ("macro_ovr_roc_auc", "Macro OvR ROC-AUC"),
               ("macro_ovr_average_precision", "Macro mean OvR average precision")]
    variants = ["baseline", "enhanced"]
    models = ["GRU", "LSTM", "LogisticRegression", "Majority baseline"]
    colors = {"baseline": "#2864a5", "enhanced": "#e07a2d"}
    for column, title in metrics:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(models)); width = .36
        for offset, variant in enumerate(variants):
            subset = selected[selected.variant.eq(variant)].set_index("model")
            values = [float(subset.loc[m, column]) for m in models]
            ax.bar(x + (offset - .5) * width, values, width, label=variant.title(), color=colors[variant])
        ax.set_xticks(x, ["GRU", "LSTM", "Logistic\nregression", "Majority"])
        ax.set_ylabel(title); ax.set_ylim(0, 1.0); ax.set_title(f"Corrected v4 common-cohort test: {title}")
        ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); fig.tight_layout()
        fig.savefig(OUT / f"controlled_{column}.png", dpi=180)
        fig.savefig(OUT / f"controlled_{column}.svg")
        plt.close(fig)

    ablation = pd.read_csv(ABL / "test_results.csv")
    feature_names = ["TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX",
                     "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(feature_names)); width = .36
    for offset, arch in enumerate(("GRU", "LSTM")):
        subset = ablation[ablation.model.eq(arch)].set_index("feature_added")
        vals = [float(subset.loc[f, "macro_f1_delta_vs_same_architecture_baseline"]) for f in feature_names]
        ax.bar(x + (offset - .5) * width, vals, width, label=arch)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(x, ["Total grouped\ncrime", "Active groups", "Diversity\nindex",
                      "Previous-year\ntotal", "YoY total\nchange"])
    ax.set_ylabel("Test macro-F1 change vs same-architecture baseline")
    ax.set_title("One engineered feature added at a time; same 671 test targets")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(OUT / "engineered_feature_macro_f1_delta.png", dpi=180)
    fig.savefig(OUT / "engineered_feature_macro_f1_delta.svg")
    plt.close(fig)
    print(f"Saved figures derived from test_results.csv to {OUT}")


if __name__ == "__main__": main()
