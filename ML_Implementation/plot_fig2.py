from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "combined_temporal_results"

# Locate the enhanced Random Forest confusion matrix
matches = list(RESULTS_DIR.rglob("confusion_matrix.csv"))
matrix_file = next(
    (
        path for path in matches
        if "enhanced" in str(path).lower()
        and "randomforest" in str(path).lower()
    ),
    None
)

if matrix_file is None:
    raise FileNotFoundError(
        "Enhanced Random Forest confusion_matrix.csv not found. "
        "Check the model output folders."
    )

cm = pd.read_csv(matrix_file, index_col=0)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True
)

plt.title("Confusion Matrix - Enhanced Random Forest")
plt.xlabel("Predicted Target Level")
plt.ylabel("Actual Target Level")
plt.tight_layout()

output_file = RESULTS_DIR / "fig2_enhanced_randomforest_confusion_matrix.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print("Graph saved to:", output_file)