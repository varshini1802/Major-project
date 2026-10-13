"""Build manuscript figures only from saved, audited experiment outputs."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "final_paper"
AUDIT = ROOT / "outputs" / "audits" / "final_corrections_v5"
V5 = ROOT / "ML_Implementation" / "corrected_temporal_results" / "corrected_v5_multiseed"
V4 = ROOT / "ML_Implementation" / "corrected_temporal_results" / "corrected_v4_controlled_sequence"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(V5 / "model_metric_summary.csv")
    summary = summary[summary.metric.eq("test_macro_f1")]
    models = ["GRU", "LSTM"]
    variants = ["baseline", "enhanced"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(models))
    width = 0.34
    colors = ["#5277a3", "#cf7951"]
    for j, variant in enumerate(variants):
        rows = summary[summary.variant.eq(variant)].set_index("model").loc[models]
        ax.bar(x + (j - 0.5) * width, rows["mean"], width,
               yerr=rows["standard_deviation"], capsize=4,
               label=variant.capitalize(), color=colors[j])
    ax.set_xticks(x, models)
    ax.set_ylabel("Locked-test macro-F1")
    ax.set_ylim(0.65, 0.75)
    ax.set_title("Five-seed baseline and enhanced sequence models")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUT / "five_seed_macro_f1.png", dpi=220)
    plt.close(fig)

    ablation = pd.read_csv(
        ROOT / "ML_Implementation" / "corrected_temporal_results"
        / "corrected_v5_feature_ablation_multiseed" / "paired_difference_summary.csv"
    )
    order = ["active crime groups", "crime diversity index", "previous-year total crime",
             "total grouped crime", "year-over-year total crime change"]
    # Match the human-readable feature labels in the run summary.
    feature_map = {
        "ACTIVE_CRIME_GROUPS": "active crime groups",
        "CRIME_DIVERSITY_INDEX": "crime diversity index",
        "PREV_YEAR_TOTAL_CRIME": "previous-year total crime",
        "TOTAL_GROUPED_CRIME": "total grouped crime",
        "YOY_TOTAL_CRIME_CHANGE": "year-over-year total crime change",
    }
    ablation["feature_label"] = ablation.feature_added.map(feature_map)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    y = np.arange(len(order))
    for offset, model, color in [(-0.12, "GRU", "#5277a3"), (0.12, "LSTM", "#cf7951")]:
        rows = ablation[ablation.model.eq(model)].set_index("feature_label").loc[order]
        ax.errorbar(rows.mean_paired_difference, y + offset,
                    xerr=rows.sd_across_seeds, fmt="o", capsize=3,
                    color=color, label=model)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlabel("Paired change in test macro-F1 (mean across five seeds; bars show seed SD)")
    ax.set_title("One-feature ablations versus the same-seed six-group baseline")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUT / "five_seed_ablation_macro_f1.png", dpi=220)
    plt.close(fig)

    # Enhanced LSTM had the largest validation macro-F1 among the six controlled
    # model/variant candidates. Selection is determined from validation only.
    selected = pd.read_csv(V4 / "validation_selection.csv")
    selected = selected.sort_values("validation_macro_f1", ascending=False).iloc[0]
    if (selected["variant"], selected["model"]) != ("enhanced", "LSTM"):
        raise RuntimeError("Saved validation selection changed; update the figure and manuscript selection.")
    cm = pd.read_csv(AUDIT / "v4_recomputed_confusion_matrices.csv")
    cm = cm[(cm.variant == "enhanced") & (cm.model == "lstm")]
    matrix = cm.pivot(index="actual_class", columns="predicted_class", values="count").loc[1:5, 1:5]
    fig, ax = plt.subplots(figsize=(5.8, 5.1))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues")
    for row in range(5):
        for col in range(5):
            val = int(matrix.iloc[row, col])
            ax.text(col, row, str(val), ha="center", va="center",
                    color="white" if val > matrix.to_numpy().max() * 0.55 else "black")
    ax.set(xticks=range(5), yticks=range(5),
           xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5],
           xlabel="Predicted burden class", ylabel="Observed burden class",
           title="Selected v4 enhanced LSTM: locked 2013–2014 test")
    fig.colorbar(image, ax=ax, label="District-year sequences")
    fig.tight_layout()
    fig.savefig(OUT / "selected_v4_enhanced_lstm_confusion_matrix.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
