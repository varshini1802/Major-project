"""Correct stale class descriptions in the observed-year notebook summary."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/observed_year_classification_models.ipynb"
start = "## 3. Target Classes"
end = "## 4. Train-Test Split"
replacement = """## 3. Target Classes and Frozen Boundaries

`TARGET_VIOLENT_LEVEL` represents five ordered levels of violent-crime burden, not five crime types. This project-defined target is an adaptation; the original paper's target-generation procedure has not been independently verified.

- Class 1: Lowest violent-crime burden
- Class 2: Low-to-moderate violent-crime burden
- Class 3: Moderate violent-crime burden
- Class 4: Moderate-to-high violent-crime burden
- Class 5: Highest violent-crime burden

The four frozen burden cut points are 107, 237, 362, and 563, fitted on 2001–2010 training observations. Equal burdens receive the same class under the saved right-sided threshold assignment rule. The canonical labels and class distributions are saved in the processed target data and audit outputs.

## 4. Observed-Year Train-Test Split"""

notebook = json.loads(PATH.read_text(encoding="utf-8"))
changed = 0
for cell in notebook.get("cells", []):
    if cell.get("cell_type") != "markdown":
        continue
    text = "".join(cell.get("source", []))
    if start not in text or end not in text:
        continue
    left = text.index(start)
    right = text.index(end, left)
    text = text[:left] + replacement + text[right + len(end):]
    cell["source"] = text.splitlines(keepends=True)
    changed += 1
if changed != 1:
    raise RuntimeError(f"Expected one notebook summary section to update, found {changed}")
PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Updated target class labels and marked the random split as observed-year classification.")
