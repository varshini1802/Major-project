# Initial status before final verification edits

Created after inspecting the requested closure documents and locating the available correction document; before making any edits for this verification turn.

## Existing evidence

- `outputs/audits/final_corrections_v5/` contains saved-v4 metric recomputation, target/feature audits, source lineage and overlap reports, schema/missingness reports, duplicate-key findings, structural forecast-availability checks, leakage negative tests, five-seed paired model results, and the five-seed ablation report/integrity checks.
- `ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/`, `corrected_v4_feature_ablation/`, and `corrected_v5_multiseed/` contain saved predictions, metrics, model configurations and/or models. The five-seed ablation outputs are in `corrected_v5_feature_ablation_multiseed/`.
- Frozen thresholds and split are documented as 107/237/362/563, train 2001–2010, validation 2011–2012, test 2013–2014. Saved-v4 metric checks pass; ablation audit reports 50/50 runs and locked-key validation.
- The closure README, report, 13-item reconstructed checklist, schema note and draft exist. The draft has methods, results and limitations, but no keywords, paper figures, or selected-model confusion matrix embedded.

## Original correction document search

- `correction(2).md` exists in `C:/Users/Dell/Downloads` and contains numbered requirements 1–13 plus paper requirements 14–20 and a final verification checklist. Its contents are directly relevant to this repository and match the correction history; it is the best available original correction document.
- No `correction.md` was present in Downloads. Repository search found no original `corrections.md`; Git history search found no file with correction/paper/conference names. Available Codex attachment directories contain pasted-text attachments, including the current final-verification prompt; they do not provide a distinct file named `correction.md` in the inspected listing. The available `correction(2).md` will be preserved as the checklist source, but the original unsuffixed file's identity/version cannot be established from repository history.

## Outstanding requirements

- Map each of the 20 requirements and final checklist items from `correction(2).md` to implemented evidence and a justified status.
- Complete the source overlap note with the nine direct columns, 76.6%/9,552 descriptive reconstructibility result, and explicit limits on interpretation.
- State missing target, missing source/group, partial aggregate and missing historical district-year as separate conditions.
- Audit every draft number, citation and claim against files; verify references and add missing required paper sections/visual evidence where saved results support it.
- Correct obsolete README/closure statements and broken/duplicated reproduction commands.

## Potential inconsistencies to resolve

- Current checklist has only 13 items and does not preserve the original paper requirements/checklist wording.
- Closure text still says the ablation results await review, although 50 runs are complete and audited.
- README lists the inventory command twice and says five-seed ablations are “when completed”.
- Data dictionary lacks the direct-source overlap list and 76.6% reconstructibility statistic explicitly.
- The draft lacks keywords and the requested figures/selected-model confusion matrix; the exact source target rule is a project adaptation and must remain qualified.
- Several documents show encoding/mojibake for en dashes/arrows; verify whether it is display-only or present in the saved text.

## Experiments not needing rerun

- Do not retrain corrected v4 or v5 full model suites. Verify existing outputs by recomputing/checking saved predictions and configs.
- Do not rerun the 50 feature-ablation fits; verify the saved summary and integrity artifacts.
- Rerun only documentation-relevant audits/tests if edits affect code or evidence indexes. Model outputs should remain byte-for-byte unchanged.
