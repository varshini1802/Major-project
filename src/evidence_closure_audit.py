"""Build evidence-closure audits without changing research inputs or results."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audits/final_evidence_closure"
V5 = ROOT / "outputs/audits/final_corrections_v5"
TABLES = ROOT / "outputs/tables"
TARGETS = ROOT / "data/processed/district_targeted.csv"
CANON = ROOT / "data/processed/district_integrated_canonical.csv"
KEYS = ["YEAR", "STATE", "UNIT_NAME"]
KNOWN_DIRECT = {
    "IPC_ATTEMPT_TO_MURDER", "IPC_CULPABLE_HOMICIDE_NOT_AMOUNTING_TO_MURDER",
    "IPC_MURDER", "IPC_KIDNAPPING_ABDUCTION", "IPC_ARSON", "IPC_RIOTS",
    "IPC_DACOITY", "IPC_ROBBERY", "IPC_RAPE",
}
OGD_CATALOG = "https://www.data.gov.in/catalog/district-wise-crimes-under-various-sections-indian-penal-code-ipc-crimes"
OGD_2013 = "https://www.data.gov.in/resource/district-wise-crime-under-various-sections-indian-penal-code-ipc-crimes-during-2013"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _date(value):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return pd.Timestamp(value).normalize()


def classify_availability(documented_release_date, forecast_origin):
    """Strict decision: catalog dates are not accepted as release dates."""
    origin = _date(forecast_origin)
    if origin is None:
        return "SCENARIO_UNDEFINED"
    release = _date(documented_release_date)
    if release is None:
        return "UNKNOWN"
    return "UNAVAILABLE_AFTER_ORIGIN" if release > origin else "AVAILABLE_BY_ORIGIN"


def calculate_overlap_reconciliation(frame: pd.DataFrame, overlap_sources: list[str]) -> dict:
    """Use the same eligible-row mask for source and burden sums."""
    required = ["TARGET_VIOLENT_LEVEL", "VIOLENT_CRIME_BURDEN", *overlap_sources]
    missing = [c for c in required if c not in frame]
    if missing:
        raise KeyError(f"Required overlap columns missing: {missing}")
    labels = pd.to_numeric(frame["TARGET_VIOLENT_LEVEL"], errors="coerce")
    burden = pd.to_numeric(frame["VIOLENT_CRIME_BURDEN"], errors="coerce")
    values = frame[overlap_sources].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    direct = values.sum(axis=1, min_count=1)
    valid = labels.notna() & burden.notna() & np.isfinite(burden) & burden.ge(0)
    observed = valid & direct.notna()
    eligible = valid & direct.notna() & burden.gt(0)
    numerator = float(direct.loc[eligible].sum())
    denominator = float(burden.loc[eligible].sum())
    all_valid_numerator = float(direct.loc[observed].sum())
    all_valid_denominator = float(burden.loc[valid].sum())
    return {
        "valid_mask": valid,
        "observed_overlap_mask": observed,
        "eligible_mask": eligible,
        "direct_values": direct,
        "numerator": numerator,
        "denominator": denominator,
        "ratio": numerator / denominator if denominator else np.nan,
        "valid_rows": int(valid.sum()),
        "eligible_rows": int(eligible.sum()),
        "valid_rows_without_overlap_values": int((valid & direct.isna()).sum()),
        "observed_overlap_rows": int(observed.sum()),
        "zero_burden_valid_rows": int((valid & burden.eq(0)).sum()),
        "zero_burden_direct_sum": float(direct.loc[valid & burden.eq(0) & direct.notna()].sum()),
        "coverage": float(observed.sum() / valid.sum()) if valid.sum() else np.nan,
        "all_valid_numerator_observed_rows_only": all_valid_numerator,
        "all_valid_burden_sum": all_valid_denominator,
        "all_valid_components_observed": bool((valid & direct.isna()).sum() == 0),
        "all_valid_ratio": all_valid_numerator / all_valid_denominator if all_valid_denominator and (valid & direct.isna()).sum() == 0 else np.nan,
    }


def protected_files() -> list[Path]:
    files = [ROOT / "src/target_creation_backup.py", ROOT / "outputs/tables/target_thresholds.csv",
             ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/locked_test_keys.csv",
             ROOT / "outputs/reports/conference_paper_draft.md"]
    files.extend(p for p in (ROOT / "data/processed").rglob("*") if p.is_file())
    exp = ROOT / "ML_Implementation/corrected_temporal_results"
    files.extend(p for p in exp.rglob("*") if p.is_file() and
                 (p.name in {"test_predictions.csv", "test_results.csv", "locked_test_keys.csv"}))
    return sorted(set(p for p in files if p.is_file()))


def current_protected_hashes() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in protected_files()}


def _write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False)


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(V5 / name)


def overlap_matrix() -> pd.DataFrame:
    source = _load("direct_indirect_source_overlap.csv")
    semantic = _load("ipc_semantic_relationship_review.csv")
    formulas = _load("group_predictor_formula_recovery.csv").set_index("predictor_group")
    relationship_sources = set(semantic["predictor_source"].astype(str))
    canon_map = pd.read_csv(TABLES / "canonical_feature_mapping.csv")
    raw_columns = set(pd.read_csv(ROOT / "data/processed/district_integrated_raw.csv", nrows=0).columns)
    rows = []
    for item in source.to_dict("records"):
        group, col = item["predictor_group"], item["predictor_source_variable"]
        is_direct = bool(item["direct_overlap"])
        if is_direct:
            overlap_type, status, risk = "DIRECT_SHARED_COMPONENT", "CONFIRMED_DIRECT", "HIGH"
        elif col in relationship_sources:
            overlap_type, status, risk = "POSSIBLE_SEMANTIC_OVERLAP", "UNRESOLVED_EXTERNAL_EVIDENCE", "POTENTIAL"
        else:
            overlap_type, status, risk = "UNRESOLVED", "UNRESOLVED_EXTERNAL_EVIDENCE", "UNKNOWN"
        formula = formulas.loc[group]
        target_groups = item.get("target_group_names", "")
        normalized = col if col in pd.read_csv(CANON, nrows=0).columns else ""
        family, _, feature = col.partition("_")
        aliases = canon_map.loc[(canon_map.crime_group == family) & (canon_map.canonical_feature == feature), "source_feature"].astype(str).tolist()
        available_aliases = [alias for alias in aliases if alias in raw_columns]
        evidence_loc = "src/target_creation.py:BASE_PAPER_GROUPS; src/grouped_dataset_integrity_audit.py; outputs/audits/final_corrections_v5/direct_indirect_source_overlap.csv"
        rows.append({
            "source_column": col, "normalized_column": normalized or "UNKNOWN_NOT_PRESENT_AS_CANONICAL_COLUMN",
            "earliest_available_raw_stage_columns": " | ".join(available_aliases),
            "raw_source_provenance": "processed integrated raw-stage alias; original download identity is unverified" if available_aliases else "UNKNOWN; no verified raw-stage alias found",
            "target_contribution": target_groups or "No exact target-source match; semantic relation remains unproven",
            "predictor_group": group, "transformation_path": f"IPC source -> {group} row sum(min_count=1) -> keyed lag or sequence history",
            "overlap_type": overlap_type, "evidence_file": evidence_loc,
            "evidence_location": f"{group}; formula={formula['source_columns']}; target groups={target_groups}",
            "risk_level": risk, "resolution_status": status,
            "direct_overlap": is_direct,
            "semantic_evidence": item.get("indirect_overlap_evidence", ""),
            "target_aggregation": "BASE_PAPER_GROUPS group sums(min_count=1), then sum(min_count=1); invalid negative row excluded",
            "predictor_aggregation": str(formula["aggregation"]) + "; " + str(formula["missing_handling"]),
            "model_scope": "tabular baseline/enhanced and recurrent sequence baseline/enhanced; same-year notebook is separate observed-year classification",
        })
    source_groups = source.groupby("predictor_source_variable").first()["predictor_group"].to_dict()
    for rel in semantic.to_dict("records"):
        col = str(rel["predictor_source"])
        if col not in source_groups:
            continue
        group = source_groups[col]
        formula = formulas.loc[group]
        classification = str(rel["relationship_class"])
        if classification == "related but distinguished offence definitions":
            overlap_type, risk = "UNRESOLVED", "UNKNOWN"
        else:
            overlap_type, risk = "POSSIBLE_SEMANTIC_OVERLAP", "POTENTIAL"
        family, _, feature = col.partition("_")
        aliases = canon_map.loc[(canon_map.crime_group == family) & (canon_map.canonical_feature == feature), "source_feature"].astype(str).tolist()
        available_aliases = [alias for alias in aliases if alias in raw_columns]
        rows.append({
            "source_column": col, "normalized_column": col if col in pd.read_csv(CANON, nrows=0).columns else "UNKNOWN_NOT_PRESENT_AS_CANONICAL_COLUMN",
            "earliest_available_raw_stage_columns": " | ".join(available_aliases),
            "raw_source_provenance": "processed integrated raw-stage alias; original download identity is unverified" if available_aliases else "UNKNOWN; no verified raw-stage alias found",
            "target_contribution": rel["target_source_or_group"], "predictor_group": group,
            "transformation_path": f"IPC source -> {group} row sum(min_count=1); relationship is a separate potential target-category relation",
            "overlap_type": overlap_type, "evidence_file": "outputs/audits/final_corrections_v5/ipc_semantic_relationship_review.csv",
            "evidence_location": f"{classification}; {rel['evidence']}; {rel['conclusion']}",
            "risk_level": risk, "resolution_status": "UNRESOLVED_EXTERNAL_EVIDENCE", "direct_overlap": bool(source.loc[source.predictor_source_variable.eq(col), "direct_overlap"].any()),
            "semantic_evidence": rel["conclusion"], "target_aggregation": "Relationship to source/group named in semantic review; no claim of shared case counts",
            "predictor_aggregation": str(formula["aggregation"]) + "; " + str(formula["missing_handling"]),
            "model_scope": "tabular baseline/enhanced and recurrent sequence baseline/enhanced; structural t-1/t-3..t-1 only",
        })
    result = pd.DataFrame(rows)
    if not KNOWN_DIRECT.issubset(set(result.loc[result.direct_overlap, "source_column"])):
        raise AssertionError("Direct overlap audit omitted a known shared source")
    return result


def overlap_ledger(matrix: pd.DataFrame) -> pd.DataFrame:
    relationships = _load("ipc_semantic_relationship_review.csv")
    rows = []
    for item in relationships.to_dict("records"):
        rows.append({
            "relationship": f"{item['predictor_source']} -> {item['target_source_or_group']}",
            "source_or_publication": item["evidence"], "publication_date": "UNKNOWN",
            "ipc_categories_and_years": "Relationship as described in repository audit; year-specific statistical crosswalk not supplied",
            "evidence_statement": item["conclusion"], "effect": "INSUFFICIENT_TO_PROVE_OR_RULE_OUT_NCRB_COUNT_OVERLAP",
            "human_domain_review_required": True, "evidence_file": "outputs/audits/final_corrections_v5/ipc_semantic_relationship_review.csv",
            "source_url": "", "resolution_status": "UNRESOLVED_EXTERNAL_EVIDENCE",
        })
    rows.append({
        "relationship": "Official year-specific NCRB category definitions/crosswalks/counting rules",
        "source_or_publication": "No applicable rule document located in repository or official catalog search",
        "publication_date": "", "ipc_categories_and_years": "Target and predictor IPC categories, 2001–2014",
        "evidence_statement": "The OGD catalog identifies the NCRB crime-in-India source and says figures are counts; it does not state parent/component or duplicate counting rules.",
        "effect": "No proof of statistical case-count overlap or independence.", "human_domain_review_required": True,
        "evidence_file": "", "source_url": OGD_CATALOG, "resolution_status": "UNRESOLVED_EXTERNAL_EVIDENCE",
    })
    return pd.DataFrame(rows)


def denominator_audit() -> pd.DataFrame:
    target = pd.read_csv(TARGETS, low_memory=False)
    mapping = pd.read_csv(TABLES / "target_predictor_source_overlap.csv")
    shared = mapping.loc[mapping["direct_overlap"].astype(str).str.lower().eq("true"), "predictor_source_variable"].drop_duplicates().tolist()
    stats = calculate_overlap_reconciliation(target, shared)
    direct = stats["direct_values"]
    burden = pd.to_numeric(target["VIOLENT_CRIME_BURDEN"], errors="coerce")
    residual = burden - direct
    eligible_residual = residual.loc[stats["eligible_mask"]]
    rows = [
        {"statistic": "Original positive-burden-row ratio", "eligibility_rule": "valid TARGET_VIOLENT_LEVEL and finite nonnegative burden; direct source sum observed; burden > 0 (prior code mask)",
         "numerator_formula": "row sum of nine unique direct IPC shared columns with skip-NA; sum over eligible rows",
         "denominator_formula": "sum VIOLENT_CRIME_BURDEN over the identical eligible rows",
         "eligible_rows": stats["eligible_rows"], "row_count": stats["valid_rows"], "numerator": stats["numerator"],
         "denominator": stats["denominator"], "value": stats["ratio"], "presentation": f"{stats['ratio']:.6%}",
         "missing_treatment": "Direct row sum min_count=1; numerator/denominator use identical positive-burden mask",
         "interpretation": "Descriptive share on positive-burden rows; 79 zero-burden rows are excluded by burden > 0, not because source values are missing", "estimability": "ESTIMABLE"},
        {"statistic": "All-valid-target-row overlap-source coverage", "eligibility_rule": "valid target label, finite burden >=0",
         "numerator_formula": "count valid target rows with at least one of nine direct sources observed",
         "denominator_formula": "count all valid target rows",
         "eligible_rows": stats["observed_overlap_rows"], "row_count": stats["valid_rows"], "numerator": stats["observed_overlap_rows"],
         "denominator": stats["valid_rows"], "value": stats["coverage"], "presentation": f"{stats['coverage']:.6%}",
         "missing_treatment": f"{stats['valid_rows_without_overlap_values']} valid rows with all nine direct components missing; zero-burden rows are observed values",
         "interpretation": "Coverage statistic, not burden share", "estimability": "ESTIMABLE"},
        {"statistic": "All-valid-target-row burden share", "eligibility_rule": "all valid target labels with finite nonnegative burden",
         "numerator_formula": "sum row-wise min_count=1 direct shared-source values over all valid rows; all valid rows have an observed direct component",
         "denominator_formula": "sum VIOLENT_CRIME_BURDEN over all valid target rows", "eligible_rows": stats["observed_overlap_rows"],
         "row_count": stats["valid_rows"], "numerator": stats["all_valid_numerator_observed_rows_only"], "denominator": stats["all_valid_burden_sum"],
         "value": stats["all_valid_ratio"], "presentation": f"{stats['all_valid_ratio']:.6%}",
         "missing_treatment": "No valid rows have all nine shared values missing; 79 zero-burden rows have observed zero shared contribution",
         "interpretation": "All-valid-row burden share; same value as positive-burden subset because omitted rows contribute zero to both sums", "estimability": "ESTIMABLE"},
        {"statistic": "Zero-burden rows omitted by legacy positive-burden mask", "eligibility_rule": "valid target label and burden == 0",
         "numerator_formula": "count rows and sum their direct-source contributions", "denominator_formula": "not a ratio",
         "eligible_rows": stats["zero_burden_valid_rows"], "row_count": stats["valid_rows"],
         "numerator": stats["zero_burden_direct_sum"], "denominator": 0, "value": np.nan,
         "presentation": f"rows={stats['zero_burden_valid_rows']}; shared source sum={stats['zero_burden_direct_sum']:.0f}",
         "missing_treatment": "Source values are observed; no missing-as-zero substitution",
         "interpretation": "Explains 9,631 versus 9,552: 79 valid rows have zero burden and are excluded by burden > 0.", "estimability": "ESTIMABLE"},
        {"statistic": "Direct-source remaining burden on positive-burden rows", "eligibility_rule": "same positive-burden mask as the original ratio",
         "numerator_formula": "sum(VIOLENT_CRIME_BURDEN - direct shared-source row sum)",
         "denominator_formula": "None; residual component count total, not a ratio", "eligible_rows": stats["eligible_rows"],
         "row_count": stats["eligible_rows"], "numerator": float(eligible_residual.sum()), "denominator": np.nan,
         "value": np.nan, "presentation": f"sum={eligible_residual.sum():.0f}; negative_rows={(eligible_residual < -1e-9).sum()}",
         "missing_treatment": "Same eligibility mask; residual is mathematically valid because direct source columns are components in target formula",
         "interpretation": "Target burden remaining after subtracting the observed direct components; confirms partial reconstructability, not full label reconstruction",
         "estimability": "ESTIMABLE"},
    ]
    result = pd.DataFrame(rows)
    if stats["valid_rows"] != 9631 or stats["eligible_rows"] != 9552 or stats["observed_overlap_rows"] != 9631:
        raise AssertionError(f"Unexpected denominator counts: valid={stats['valid_rows']}, positive={stats['eligible_rows']}, observed={stats['observed_overlap_rows']}")
    if stats["valid_rows_without_overlap_values"] != 0 or stats["zero_burden_valid_rows"] != 79 or stats["zero_burden_direct_sum"] != 0:
        raise AssertionError("Expected 79 observed zero-burden rows and no valid rows missing direct-source values")
    if not np.isclose(stats["numerator"], 3228843.0) or not np.isclose(stats["denominator"], 4215620.0):
        raise AssertionError("Direct-overlap sums differ from frozen audited values")
    return result


def availability_ledger() -> pd.DataFrame:
    features = ["GROUP_HOMICIDE", "GROUP_SEXUAL_VIOLENCE", "GROUP_KIDNAPPING_ABDUCTION", "GROUP_ROBBERY_DACOITY",
                "GROUP_PUBLIC_VIOLENCE", "GROUP_PROPERTY_CRIME", "TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS",
                "CRIME_DIVERSITY_INDEX", "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"]
    rows = []
    def resource_for(source_year):
        years = [int(x) for x in re.findall(r"\d{4}", str(source_year))]
        if years and max(years) <= 2012 and min(years) >= 2001:
            return "OGD IPC district-wise resource covering 2001–2012", "2013-09-17", OGD_CATALOG
        if years == [2013]:
            return "OGD IPC district-wise resource during 2013", "2014-10-16", OGD_2013
        return "No matching year-specific OGD resource established for this source year", "", ""

    def add_record(feature, target_year, mode, feature_row_year, source_year, operation):
        resource, catalog_date, url = resource_for(source_year)
        rows.append({
            "feature": feature, "target_prediction_year": target_year, "model_input_path": mode,
            "feature_observation_year": feature_row_year, "source_resource": resource,
            "data_reference_year": source_year, "catalog_publication_date": catalog_date,
            "documented_release_date": "", "documented_retrieval_date": "",
            "forecast_origin": "UNDEFINED_IN_PROJECT", "availability_evidence": f"OGD catalog metadata only; associated resource reference {url or 'not identified'}; actual NCRB release/retrieval not established",
            "documented_release_status": "UNKNOWN",
            "availability_status": classify_availability(None, None), "release_vs_origin_strict_status": classify_availability(None, None),
            "earliest_evidenced_public_availability": "UNKNOWN; catalog publication date is not treated as actual release date",
            "evidence_url_or_file": url, "transformation_path": operation,
            "limitations": "Catalog publication date is separate from actual release, retrieval, and forecaster availability; origin is undefined.",
        })

    # Tabular artifact target years with a keyed prior-year row.
    for target_year in range(2002, 2015):
        obs_year = target_year - 1
        for feature in features:
            if feature == "PREV_YEAR_TOTAL_CRIME":
                add_record(feature, target_year, "tabular_t-1", obs_year, obs_year - 1,
                           "PREV_YEAR_TOTAL_CRIME at feature year t-1 reads same district aggregate from t-2")
            elif feature == "YOY_TOTAL_CRIME_CHANGE":
                for source_year in (obs_year - 1, obs_year):
                    add_record(feature, target_year, "tabular_t-1", obs_year, source_year,
                               "YOY at feature year t-1 subtracts t-2 grouped total from t-1 grouped total")
            else:
                add_record(feature, target_year, "tabular_t-1", obs_year, obs_year,
                           "t-1 grouped source row; derived features aggregate the six groups from this row")

    # Recurrent arrays use three consecutive feature years for each target.
    for target_year in range(2004, 2015):
        for obs_year in range(target_year - 3, target_year):
            for feature in features:
                if feature == "PREV_YEAR_TOTAL_CRIME":
                    add_record(feature, target_year, f"sequence_input_{obs_year}", obs_year, obs_year - 1,
                               "PREV_YEAR_TOTAL_CRIME in sequence row r uses same district aggregate at r-1")
                elif feature == "YOY_TOTAL_CRIME_CHANGE":
                    for source_year in (obs_year - 1, obs_year):
                        add_record(feature, target_year, f"sequence_input_{obs_year}", obs_year, source_year,
                                   "YOY in sequence row r compares grouped totals at r and r-1")
                else:
                    add_record(feature, target_year, f"sequence_input_{obs_year}", obs_year, obs_year,
                               "same-district historical sequence row; source year equals input observation year")
    return pd.DataFrame(rows)


def source_manifest() -> pd.DataFrame:
    rows = []
    clean = ROOT / "data/processed"
    for path in sorted(clean.rglob("*.csv")):
        year_value = "UNKNOWN"
        try:
            columns = pd.read_csv(path, nrows=0).columns
            year_col = next((c for c in ("YEAR", "Year", "year") if c in columns), None)
            if year_col:
                years = pd.to_numeric(pd.read_csv(path, usecols=[year_col])[year_col], errors="coerce").dropna()
                if not years.empty:
                    year_value = f"{int(years.min())}-{int(years.max())}" if years.min() != years.max() else str(int(years.iloc[0]))
        except Exception:
            year_value = "UNKNOWN"
        if year_value == "UNKNOWN":
            named_years = re.findall(r"20\d{2}", path.name)
            if named_years:
                year_value = "-".join(named_years)
        is_raw_stage = "integrated_raw" in path.name
        rows.append({
            "dataset_or_file": path.relative_to(ROOT).as_posix(), "source_year": year_value,
            "source_url_or_identifier": "UNKNOWN; local file provenance to OGD/NCRB resource not proven",
            "original_filename": "UNKNOWN_NOT_PROVEN_ORIGINAL_DOWNLOAD", "download_date": "UNKNOWN", "catalog_publication_date": "",
            "sha256": sha256(path), "transformation_script": "src/data_loading.py; src/data_cleaning.py; notebooks/01_source_data_integration.ipynb (stage-specific association; exact producing run not logged)",
            "output_file": path.relative_to(ROOT).as_posix(),
            "provenance_status": "PROCESSED_STAGE_HASHED; UPSTREAM_CHAIN_UNVERIFIED" if not is_raw_stage else "PROCESSED_RAW_STAGE_HASHED; ORIGINAL_SOURCE_LINK_UNVERIFIED",
            "missing_evidence": "Original downloaded file, source URL/identifier tied to local file, retrieval date, and archived source hash.",
        })
    for name in ("src/data_loading.py", "src/data_cleaning.py", "notebooks/01_source_data_integration.ipynb",
                 "src/target_creation.py", "src/feature_engineering.py", "ML_Implementation/forecast_feature_contract.py",
                 "ML_Implementation/sequence_generation.py"):
        path = ROOT / name
        rows.append({"dataset_or_file": name, "source_year": "N/A", "source_url_or_identifier": "N/A code", "original_filename": "N/A",
                     "download_date": "N/A", "catalog_publication_date": "N/A", "sha256": sha256(path), "transformation_script": name,
                     "output_file": "N/A", "provenance_status": "LOCAL_CODE_HASHED", "missing_evidence": "Execution manifest linking exact input hashes to producing run is not complete."})
    rows.extend([
        {"dataset_or_file": "OGD/NCRB district IPC resource catalog record, coverage 2001–2012", "source_year": "2001-2012",
         "source_url_or_identifier": OGD_CATALOG, "original_filename": "UNKNOWN", "download_date": "UNKNOWN",
         "catalog_publication_date": "2013-09-17", "sha256": "", "transformation_script": "UNKNOWN",
         "output_file": "UNKNOWN; no verified local-file linkage", "provenance_status": "OFFICIAL_CATALOG_RECORD_FOUND_LOCAL_LINK_UNVERIFIED",
         "missing_evidence": "Resource file identity, download URL/hash, retrieval timestamp, and processing-run manifest."},
        {"dataset_or_file": "OGD/NCRB district IPC resource catalog record, 2013", "source_year": "2013",
         "source_url_or_identifier": OGD_2013, "original_filename": "UNKNOWN", "download_date": "UNKNOWN",
         "catalog_publication_date": "2014-10-16", "sha256": "", "transformation_script": "UNKNOWN",
         "output_file": "UNKNOWN; no verified local-file linkage", "provenance_status": "OFFICIAL_CATALOG_RECORD_FOUND_LOCAL_LINK_UNVERIFIED",
         "missing_evidence": "Resource file identity, download URL/hash, retrieval timestamp, and processing-run manifest."},
        {"dataset_or_file": "OGD/NCRB district IPC resource catalog record, 2014", "source_year": "2014",
         "source_url_or_identifier": OGD_CATALOG, "original_filename": "UNKNOWN", "download_date": "UNKNOWN",
         "catalog_publication_date": "2016-08-18", "sha256": "", "transformation_script": "UNKNOWN",
         "output_file": "UNKNOWN; not used as t-1 predictor for the locked 2014 forecast", "provenance_status": "OFFICIAL_CATALOG_RECORD_FOUND_LOCAL_LINK_UNVERIFIED",
         "missing_evidence": "Exact resource URL/file identity, download/retrieval timestamp, and processing-run manifest."},
    ])
    return pd.DataFrame(rows)


def duplicate_audit() -> pd.DataFrame:
    clean_dir = ROOT / "data/processed/district_cleaned"
    core_dir = ROOT / "data/processed/district_core"
    file_names = [
        "01_District_wise_crimes_committed_IPC_2001_2012.csv",
        "02_01_District_wise_crimes_committed_against_SC_2001_2012.csv",
        "02_District_wise_crimes_committed_against_ST_2001_2012.csv",
        "03_District_wise_crimes_committed_against_children_2001_2012.csv",
        "42_District_wise_crimes_committed_against_women_2001_2012.csv",
    ]
    generated_metadata = {"SOURCE_FILE", "SOURCE_ROW_ID", "SOURCE_RECORD_ID", "DUPLICATE_KEY_FLAG",
                          "DATA_QUALITY_FLAG", "UNIT_TYPE", "SOURCE_QUALITY_STATUS"}
    records = []
    for filename in file_names:
        data = pd.read_csv(clean_dir / filename, low_memory=False)
        if "DISTRICT" in data and "UNIT_NAME" not in data:
            data = data.rename(columns={"DISTRICT": "UNIT_NAME"})
        data["YEAR"] = pd.to_numeric(data["YEAR"], errors="raise").astype(int)
        for col in ("STATE", "UNIT_NAME"):
            data[col] = data[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
        core_path = core_dir / filename
        core = pd.read_csv(core_path, low_memory=False) if core_path.exists() else pd.DataFrame()
        if not core.empty:
            if "DISTRICT" in core and "UNIT_NAME" not in core:
                core = core.rename(columns={"DISTRICT": "UNIT_NAME"})
            core["YEAR"] = pd.to_numeric(core["YEAR"], errors="raise").astype(int)
            for col in ("STATE", "UNIT_NAME"):
                core[col] = core[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
        payload_columns = [c for c in data.columns if c not in KEYS and c not in generated_metadata]
        all_nonkey_columns = [c for c in data.columns if c not in KEYS]
        repeated = data[data.duplicated(KEYS, keep=False)].sort_values(KEYS)
        for key, group in repeated.groupby(KEYS, dropna=False, sort=True):
            varying_payload = [c for c in payload_columns if group[c].nunique(dropna=False) > 1]
            varying_any = [c for c in all_nonkey_columns if group[c].nunique(dropna=False) > 1]
            identical = not varying_payload
            core_match = False
            if not core.empty:
                core_match = bool(((core.YEAR == key[0]) & (core.STATE == key[1]) & (core.UNIT_NAME == key[2])).any())
            records.append({
                "file": filename, "year": key[0], "state": key[1], "unit_name": key[2],
                "copy_count": len(group),
                "classification": "IDENTICAL_OFFENCE_PAYLOAD_DISTINCT_PIPELINE_ROW_IDS" if identical else "CONFLICTING_OFFENCE_PAYLOAD",
                "offence_payload_identical": identical, "varying_offence_columns": "|".join(varying_payload),
                "varying_all_nonkey_columns": "|".join(varying_any),
                "source_row_ids": "|".join(group.get("SOURCE_ROW_ID", pd.Series("", index=group.index)).astype(str)),
                "source_quality_flags": "|".join(group.get("SOURCE_QUALITY_STATUS", pd.Series("", index=group.index)).astype(str)),
                "unit_type": str(group["UNIT_TYPE"].iloc[0]) if "UNIT_TYPE" in group else "unknown",
                "present_in_core_stage": core_match,
                "key_impact": "present in core; inspect further" if core_match else "excluded from core by duplicate quality rule",
            })
    detail = pd.DataFrame(records)
    if len(detail) != 7 or detail.copy_count.sum() != 14:
        raise AssertionError(f"Recomputed duplicate groups/rows differ: groups={len(detail)} rows={detail.copy_count.sum()}")
    old = _load("cleaned_duplicate_key_records.csv")
    new_key = detail.set_index(["file", "year", "state", "unit_name"]).sort_index()
    old_key = old.set_index(["file", "year", "state", "unit_name"]).sort_index()
    left = new_key[["copy_count", "classification", "varying_offence_columns"]].copy()
    right = old_key[["copy_count", "classification", "varying_offence_columns"]].copy()
    left["varying_offence_columns"] = left["varying_offence_columns"].fillna("")
    right["varying_offence_columns"] = right["varying_offence_columns"].fillna("")
    pd.testing.assert_frame_equal(left, right, check_dtype=False)
    detail["original_identifiers"] = "UNKNOWN; SOURCE_ROW_ID/SOURCE_RECORD_ID are pipeline metadata, not proven NCRB IDs"
    detail["documented_transformation_explains_duplication"] = "NO; exclusion is documented, cause of duplicates is not"
    detail["analytical_treatment"] = detail["key_impact"]
    detail["resolution_status"] = np.where(detail["classification"].eq("CONFLICTING_OFFENCE_PAYLOAD"),
                                            "UNRESOLVED_EXTERNAL_SOURCE_IDENTIFIERS", "EXCLUDED_BY_DOCUMENTED_DUPLICATE_QUALITY_RULE")
    detail["evidence_needed"] = np.where(detail["classification"].eq("CONFLICTING_OFFENCE_PAYLOAD"),
                                         "Original NCRB extract and unique Railway unit identifier/crosswalk for 2010",
                                         "Original source record IDs or upstream extraction log to establish whether duplicate records originate upstream")
    return detail.rename(columns={"year": "key_year", "state": "key_state", "unit_name": "key_unit_name"})


def missingness_audit() -> pd.DataFrame:
    canonical = pd.read_csv(CANON, low_memory=False)
    target_map = pd.read_csv(TABLES / "target_predictor_source_overlap.csv")
    source_cols = sorted(set(target_map["predictor_source_variable"].astype(str)))
    target_sources = pd.read_csv(TABLES / "target_source_column_mapping.csv")["source_column"].astype(str).tolist()
    source_cols = sorted(set(source_cols) | set(target_sources))
    rows = []
    for col in source_cols:
        for year, idx in canonical.groupby("YEAR").groups.items():
            present = col in canonical.columns
            vals = pd.to_numeric(canonical.loc[idx, col], errors="coerce") if present else pd.Series(np.nan, index=idx)
            rows.append({"source_column": col, "year": int(year), "rows": int(len(idx)), "observed": int(vals.notna().sum()),
                         "observed_zero": int(vals.eq(0).sum()), "observed_positive": int(vals.gt(0).sum()),
                         "observed_negative": int(vals.lt(0).sum()), "missing": int(vals.isna().sum()), "column_present": present,
                         "variable": col, "missing_reason": "UNKNOWN_CAUSE" if present else "UNAVAILABLE_SOURCE_COLUMN",
                         "structural_missingness_status": "NOT_IDENTIFIABLE_FROM_VALUE_ALONE",
                         "documented_reason_counts": 0,
                         "source_category_availability": "COLUMN_PRESENT; CELL_MISSINGNESS_COUNTS_REPORTED" if present else "SOURCE_COLUMN_ABSENT_FROM_CANONICAL_SCHEMA",
                         "source_file": CANON.relative_to(ROOT).as_posix()})
    targeted = pd.read_csv(TARGETS, low_memory=False)
    for col in ("VIOLENT_CRIME_BURDEN", "TARGET_VIOLENT_LEVEL"):
        for year, idx in targeted.groupby("YEAR").groups.items():
            vals = pd.to_numeric(targeted.loc[idx, col], errors="coerce")
            rows.append({"source_column": col, "year": int(year), "rows": int(len(idx)), "observed": int(vals.notna().sum()),
                         "observed_zero": int(vals.eq(0).sum()), "observed_positive": int(vals.gt(0).sum()),
                         "observed_negative": int(vals.lt(0).sum()), "missing": int(vals.isna().sum()), "column_present": True,
                         "variable": col, "missing_reason": "TARGET_UNAVAILABLE; SPECIFIC_SOURCE_CAUSE_UNKNOWN" if col == "TARGET_VIOLENT_LEVEL" else "BURDEN_UNAVAILABLE; SOURCE_CAUSE_UNKNOWN",
                         "structural_missingness_status": "TARGET_RULE_DEPENDENT; SOURCE_CAUSE_UNKNOWN",
                         "documented_reason_counts": 0, "source_category_availability": "derived canonical target output",
                         "source_file": TARGETS.relative_to(ROOT).as_posix()})
    grouped = pd.read_csv(ROOT / "data/processed/district_grouped_reduced.csv")
    enhanced_for_history = pd.read_csv(ROOT / "data/processed/district_enhanced_features.csv")
    group_cols = [c for c in grouped.columns if c.startswith("GROUP_")]
    group_total = grouped[group_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    history = grouped[KEYS].copy()
    history["PREV_KEY_YEAR"] = pd.to_numeric(history["YEAR"], errors="coerce") + 1
    history["PRIOR_GROUPED_TOTAL"] = group_total
    history = history.rename(columns={"YEAR": "SOURCE_YEAR"}).rename(columns={"PREV_KEY_YEAR": "YEAR"})
    history["PRIOR_KEY_PRESENT"] = True
    hist_join = enhanced_for_history[KEYS].merge(history[KEYS + ["PRIOR_GROUPED_TOTAL", "PRIOR_KEY_PRESENT"]],
                                                   on=KEYS, how="left", validate="one_to_one")
    history_stats = {}
    for year, idx in enhanced_for_history.groupby("YEAR").groups.items():
        sub = enhanced_for_history.loc[idx]
        joined = hist_join.loc[idx]
        previous_missing_key = joined["PRIOR_KEY_PRESENT"].isna()
        prior_total_missing = joined["PRIOR_KEY_PRESENT"].notna() & joined["PRIOR_GROUPED_TOTAL"].isna()
        current_total = sub[group_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        previous_total_missing_feature = pd.to_numeric(sub.get("PREV_YEAR_TOTAL_CRIME"), errors="coerce").isna()
        yoy_missing = pd.to_numeric(sub.get("YOY_TOTAL_CRIME_CHANGE"), errors="coerce").isna()
        history_stats[int(year)] = {
            "missing_previous_year_observation_count": int(previous_missing_key.sum()),
            "prior_key_present_but_total_missing_count": int(prior_total_missing.sum()),
            "previous_total_feature_missing_count": int(previous_total_missing_feature.sum()),
            "yoy_missing_count": int(yoy_missing.sum()),
            "yoy_missing_due_current_total_count": int((yoy_missing & current_total.isna()).sum()),
            "yoy_missing_due_previous_total_count": int((yoy_missing & pd.to_numeric(sub.get("PREV_YEAR_TOTAL_CRIME"), errors="coerce").isna()).sum()),
        }
    for file, variables in [("data/processed/district_grouped_reduced.csv", grouped),
                            ("data/processed/district_enhanced_features.csv", pd.read_csv(ROOT / "data/processed/district_enhanced_features.csv"))]:
        frame = variables
        candidates = [c for c in frame.columns if c in {
            "GROUP_HOMICIDE", "GROUP_SEXUAL_VIOLENCE", "GROUP_KIDNAPPING_ABDUCTION", "GROUP_ROBBERY_DACOITY",
            "GROUP_PUBLIC_VIOLENCE", "GROUP_PROPERTY_CRIME", "TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS",
            "CRIME_DIVERSITY_INDEX", "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"}]
        for col in candidates:
            numeric = pd.to_numeric(frame[col], errors="coerce")
            for year, idx in frame.groupby("YEAR").groups.items():
                vals = numeric.loc[idx]
                reason_counts = history_stats.get(int(year), {}) if file.endswith("district_enhanced_features.csv") else {}
                missing_reason = "UNKNOWN_CAUSE"
                structural_status = "UNKNOWN_CAUSE_NOT_IDENTIFIABLE_FROM_VALUE_ALONE"
                if col == "PREV_YEAR_TOTAL_CRIME":
                    missing_reason = "MISSING_PREVIOUS_YEAR_OBSERVATION_OR_PRIOR_TOTAL_UNAVAILABLE"
                    structural_status = "EXACT_KEYED_PRIOR_YEAR_GAP_COUNTS_REPORTED"
                elif col == "YOY_TOTAL_CRIME_CHANGE":
                    missing_reason = "CURRENT_OR_PREVIOUS_GROUPED_TOTAL_UNAVAILABLE"
                    structural_status = "DEPENDENT_INPUT_MISSINGNESS_COUNTS_REPORTED"
                rows.append({"source_column": col, "year": int(year), "rows": int(len(idx)), "observed": int(vals.notna().sum()),
                             "observed_zero": int(vals.eq(0).sum()), "missing": int(vals.isna().sum()), "column_present": True,
                             "variable": col, "missing_reason": missing_reason,
                             "structural_missingness_status": structural_status,
                             "documented_reason_counts": 0, "source_category_availability": f"derived feature; source file {file}",
                             "observed_positive": int(vals.gt(0).sum()), "observed_negative": int(vals.lt(0).sum()),
                             "missing_reason_counts": json.dumps(reason_counts),
                             "source_file": file})
    out = pd.DataFrame(rows)
    out["observed_positive"] = out.get("observed_positive", pd.Series(index=out.index, dtype=float))
    out["observed_positive"] = out["observed_positive"].fillna(0)
    out["observed_negative"] = out.get("observed_negative", pd.Series(index=out.index, dtype=float)).fillna(0)
    out["documented_reason_counts"] = out["documented_reason_counts"].fillna(0)
    # Add explicit schema/unmapped categories from the source audit.
    discrepancies = _load("ipc_group_source_schema_discrepancies.csv")
    for item in discrepancies.to_dict("records"):
        out.loc[len(out)] = {"source_column": item.get("source_column", "UNKNOWN"), "year": item.get("year", "UNKNOWN"),
                             "rows": 0, "observed": 0, "observed_zero": 0, "missing": 0, "column_present": False,
                             "variable": item.get("source_column", "UNKNOWN"), "missing_reason": "UNMAPPED_CATEGORY_OR_SCHEMA_MISMATCH",
                             "structural_missingness_status": "SCHEMA_MAPPING_UNRESOLVED", "documented_reason_counts": 0,
                             "source_category_availability": str(item), "observed_positive": 0, "observed_negative": 0,
                             "source_file": "outputs/audits/final_corrections_v5/ipc_group_source_schema_discrepancies.csv"}
    duplicates = _load("cleaned_duplicate_key_records.csv")
    for item in duplicates.to_dict("records"):
        out.loc[len(out)] = {"source_column": "DUPLICATE_DISTRICT_YEAR_RECORD", "year": item["year"],
                             "rows": int(item["copy_count"]), "observed": 0, "observed_zero": 0, "missing": pd.NA,
                             "column_present": pd.NA, "variable": "DISTRICT_YEAR_RECORD", "missing_reason": "DUPLICATE_DISTRICT_YEAR_EXCLUDED",
                             "structural_missingness_status": "RECORD_EXCLUSION_NOT_CELL_MISSINGNESS",
                             "documented_reason_counts": 0, "source_category_availability": item["classification"],
                             "observed_positive": 0, "observed_negative": 0,
                             "missing_reason_counts": "", "source_file": item["file"]}
    return out


def integration_audit() -> dict:
    frames = []
    for path in sorted((ROOT / "data/processed").rglob("*.csv")):
        if path.parent.name not in {"district_cleaned", "district_core", "district_normalized"}:
            continue
        df = pd.read_csv(path, low_memory=False)
        if all(k in df for k in ["YEAR", "STATE"]) and ("UNIT_NAME" in df or "DISTRICT" in df):
            work = df.copy()
            if "UNIT_NAME" not in work and "DISTRICT" in work:
                work = work.rename(columns={"DISTRICT": "UNIT_NAME"})
            work["YEAR"] = pd.to_numeric(work["YEAR"], errors="coerce")
            for k in ["STATE", "UNIT_NAME"]:
                work[k] = work[k].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
            dups = int(work.duplicated(KEYS, keep=False).groupby(work[KEYS].apply(tuple, axis=1)).any().sum()) if False else int(work.duplicated(KEYS, keep=False).sum())
            frames.append({"dataset_or_file": path.relative_to(ROOT).as_posix(), "rows": len(df), "key_columns": "YEAR|STATE|UNIT_NAME",
                           "duplicate_key_rows": dups, "unique_key_count": int(work[KEYS].drop_duplicates().shape[0]),
                           "assessment": "duplicates present; see group-level resolution audit" if dups else "unique at normalized district-year key"})
    raw_final_rows = []
    raw_path = ROOT / "data/processed/district_integrated_raw.csv"
    final_path = ROOT / "data/processed/district_integrated_final.csv"
    mapping_path = TABLES / "canonical_feature_mapping.csv"
    raw = pd.read_csv(raw_path, low_memory=False)
    final = pd.read_csv(final_path, low_memory=False)
    mapping = pd.read_csv(mapping_path)
    def normalize_key_frame(frame):
        data = frame.copy()
        if "UNIT_NAME" not in data and "DISTRICT" in data:
            data = data.rename(columns={"DISTRICT": "UNIT_NAME"})
        data["YEAR"] = pd.to_numeric(data["YEAR"], errors="coerce")
        for key in ("STATE", "UNIT_NAME"):
            data[key] = data[key].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
        return data
    raw = normalize_key_frame(raw)
    final = normalize_key_frame(final)
    if raw.duplicated(KEYS).any() or final.duplicated(KEYS).any():
        raise AssertionError("Integrated raw/final key duplication detected")
    raw_i, final_i = raw.set_index(KEYS), final.set_index(KEYS)
    key_equal = raw_i.index.equals(final_i.index)
    for (_, _), group in mapping.groupby(["crime_group", "canonical_feature"], sort=False):
        output = f"{group.crime_group.iloc[0]}_{group.canonical_feature.iloc[0]}"
        if not output.startswith("IPC_") or output not in final_i.columns:
            continue
        aliases = [c for c in group.source_feature.astype(str) if c in raw_i.columns]
        if not aliases:
            raw_final_rows.append({"comparison": "raw_aliases_to_final", "variable": output, "source_aliases": "",
                                   "rows": len(final), "key_sets_match": key_equal, "value_mismatches": "",
                                   "result": "UNMAPPED_SOURCE_ALIAS"})
            continue
        reconstructed = raw_i[aliases].bfill(axis=1).iloc[:, 0]
        left = pd.to_numeric(reconstructed, errors="coerce")
        right = pd.to_numeric(final_i[output], errors="coerce")
        mismatch = int((~((left.eq(right)) | (left.isna() & right.isna()))).sum())
        raw_final_rows.append({"comparison": "raw_aliases_to_final", "variable": output, "source_aliases": " | ".join(aliases),
                               "rows": len(final), "key_sets_match": key_equal, "value_mismatches": mismatch,
                               "result": "PASS" if key_equal and mismatch == 0 else "MISMATCH"})
    for filename in ["01_District_wise_crimes_committed_IPC_2014.csv"]:
        normalized = pd.read_csv(ROOT / "data/processed/district_normalized" / filename, low_memory=False)
        cleaned = pd.read_csv(ROOT / "data/processed/district_cleaned" / filename, low_memory=False)
        core = pd.read_csv(ROOT / "data/processed/district_core" / filename, low_memory=False)
        for stage_name, frame in [("normalized_to_cleaned", cleaned), ("cleaned_to_core", core)]:
            left = normalize_key_frame(normalized if stage_name == "normalized_to_cleaned" else cleaned).set_index(KEYS)
            right = normalize_key_frame(frame).set_index(KEYS)
            common = left.index.intersection(right.index)
            compare_cols = [c for c in ("Murder", "Rape", "Dacoity", "Robbery", "Theft", "Riots", "Cheating", "Arson") if c in left and c in right]
            mismatch = sum(int((~((pd.to_numeric(left.loc[common, c], errors="coerce").eq(pd.to_numeric(right.loc[common, c], errors="coerce"))) |
                                      (pd.to_numeric(left.loc[common, c], errors="coerce").isna() & pd.to_numeric(right.loc[common, c], errors="coerce").isna()))).sum()) for c in compare_cols)
            raw_final_rows.append({"comparison": stage_name, "variable": "|".join(compare_cols), "source_aliases": filename,
                                   "rows": len(common), "key_sets_match": left.index.equals(right.index), "value_mismatches": mismatch,
                                   "result": "PASS_VALUE_SUBSET" if mismatch == 0 else "MISMATCH"})
    return {"stages": pd.DataFrame(frames), "reconciliation": pd.DataFrame(raw_final_rows),
            "key_summary": _load("district_year_key_validation.csv")}


def write_initial_status() -> None:
    path = OUT / "initial_git_status.txt"
    if path.exists():
        return
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    path.write_text(status, encoding="utf-8")


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_initial_status()
    before_path = OUT / "protected_input_hashes_before.json"
    if not before_path.exists():
        before_path.write_text(json.dumps(current_protected_hashes(), indent=2) + "\n", encoding="utf-8")
    before = json.loads(before_path.read_text(encoding="utf-8"))

    matrix = overlap_matrix()
    _write(matrix, "target_predictor_overlap_matrix.csv")
    _write(overlap_ledger(matrix), "overlap_evidence_ledger.csv")
    _write(availability_ledger(), "forecast_availability_ledger.csv")
    _write(source_manifest(), "source_provenance_manifest.csv")
    _write(duplicate_audit(), "duplicate_key_resolution_audit.csv")
    _write(missingness_audit(), "missingness_semantics_audit.csv")
    denominator = denominator_audit()
    _write(denominator, "overlap_denominator_reconciliation.csv")
    integration = integration_audit()
    _write(integration["stages"], "district_year_stage_recheck.csv")
    _write(integration["reconciliation"], "integration_reconciliation_recheck.csv")
    write_denominator_note(denominator)
    write_report(matrix, denominator)
    write_status()
    write_changed_files()
    after = current_protected_hashes()
    protected_unchanged = before == after
    if not protected_unchanged:
        changed = sorted(set(before) | set(after))
        changed = [p for p in changed if before.get(p) != after.get(p)]
        raise AssertionError(f"Protected inputs changed during audit: {changed}")
    write_manifest(before, after)


def write_status() -> None:
    rows = [
        ("OVL-1", "Direct shared target-predictor components", "COMPLETE", "target_predictor_overlap_matrix.csv; overlap_denominator_reconciliation.csv", "Nine IPC source columns in five group predictors; baseline/enhanced tabular and sequence families", "None for direct source identity", "None"),
        ("OVL-2", "Indirect statistical case-count overlap", "BLOCKED_EXTERNAL_EVIDENCE", "overlap_evidence_ledger.csv; outputs/audits/final_corrections_v5/ipc_semantic_relationship_review.csv", "Potential theft/robbery/dacoity, parent/subhead, cross-domain relationships", "Obtain NCRB year-specific category crosswalk and tabulation/counting rules for 2001–2014; human domain review", "Official NCRB schema and event/category counting documentation"),
        ("OVL-3", "Partial target reconstructability", "PARTIAL", "overlap_denominator_reconciliation.csv", "Observed direct shared components cover a portion of burden on rows with observed components", "Review residual/source categories with NCRB counting documentation; no model sensitivity retrained", "External category/accounting rules"),
        ("AV-1", "Structural historical alignment", "COMPLETE", "outputs/audits/final_closure/forecast_integrity_audit.csv; tests/test_final_evidence_closure.py", "Tabular t−1 and recurrent t−3..t−1 input construction", "None for code-level alignment", "None"),
        ("AV-2", "Historical operational publication availability", "BLOCKED_EXTERNAL_EVIDENCE", "forecast_availability_ledger.csv", "Annual sources 2001–2013 used for target years 2002–2014", "Define forecast origin; obtain NCRB release/issue records and historical retrieval/access evidence", "Forecast decision date and official release/archive timestamps; OGD catalog dates alone are insufficient"),
        ("PR-1", "Upstream file provenance", "BLOCKED_EXTERNAL_EVIDENCE", "source_provenance_manifest.csv", "Processed and normalized files in data/processed", "Provide original downloads and source catalog identifiers, retrieval dates, and extraction manifests", "Original source files and download records"),
        ("PR-2", "Duplicate normalized district-year groups", "PARTIAL", "duplicate_key_resolution_audit.csv", "Seven duplicate groups across five cleaned files, including two 2010 Railway conflicts", "Provide original NCRB rows and stable unit identifiers; existing duplicates remain excluded", "Original source identifiers and Railway unit crosswalk"),
        ("PR-3", "Administrative boundary continuity", "BLOCKED_EXTERNAL_EVIDENCE", "district_year_stage_recheck.csv; outputs/audits/final_corrections_v5/district_year_key_validation.csv", "District labels across 2001–2014; OGD caveat notes police districts and special units", "Obtain year-specific district/unit codebooks and merger/split crosswalks", "Official administrative boundary histories and stable unit identifiers"),
        ("MS-1", "Missingness counts and aggregation semantics", "COMPLETE", "missingness_semantics_audit.csv; src/feature_engineering.py; tests/test_final_evidence_closure.py", "IPC components, six grouped values, engineered features by available year", "None for counts/formula semantics", "None"),
        ("MS-2", "Causes of missing and unavailable values", "BLOCKED_EXTERNAL_EVIDENCE", "missingness_semantics_audit.csv", "Missing IPC cells, prior-year gaps, schema mismatches, excluded duplicate records", "Provide source reporting notes, original forms or extraction logs that identify causes", "Official reporting/missingness documentation and original records"),
        ("DEN-1", "Overlap statistic denominator and coverage", "COMPLETE", "overlap_denominator_reconciliation.csv; denominator_correction_note.md", "9,631 valid rows have observed direct components; 9,552 are positive-burden rows; 79 observed zero-burden rows are excluded by the prior burden > 0 mask", "Use same-row sums and state the exact mask; correct the prior 79-row missing-source interpretation in a later author revision", "None for audit calculation"),
        ("DEN-2", "All-valid-row burden share", "COMPLETE", "overlap_denominator_reconciliation.csv", "All 9,631 valid rows have a direct source observation; 79 have zero burden and zero direct contribution", "Report the all-valid ratio or explain why it equals the positive-burden subset ratio", "None for the current computed source values"),
    ]
    pd.DataFrame(rows, columns=["issue_id", "issue", "status", "evidence", "affected_scope", "next_action", "external_dependency"]).to_csv(OUT / "closure_status.csv", index=False)


def write_denominator_note(df: pd.DataFrame) -> None:
    p = df.set_index("statistic")
    ratio = p.loc["Original positive-burden-row ratio"]
    cov = p.loc["All-valid-target-row overlap-source coverage"]
    text = f"""# Denominator correction note

This note is an audit artifact for the author. The paper draft was not changed.

## What the reproducible ratio measures

The prior code's positive-burden-row ratio is **{ratio['numerator']:,.0f} / {ratio['denominator']:,.0f} = {float(ratio['value']):.6%}**. The code uses **{int(ratio['eligible_rows']):,}** rows: valid target and finite nonnegative burden, observed direct-source sum, and burden greater than zero. These numerator and denominator sums use the same row mask.

This is a descriptive shared-component burden share over positive-burden rows. The all-valid-row calculation is also estimable and has the same ratio because the 79 excluded valid rows have zero burden and observed zero shared-source contribution. It does not demonstrate that every label can be reconstructed from the predictor subset.

## Why 9,552 differs from 9,631

There are 9,631 valid target rows. All 9,631 have at least one observed value among the nine shared source columns. The legacy 9,552 count is the number of positive-burden rows because the original code also requires `burden > 0`; the remaining 79 rows have zero burden and observed zero direct-source contribution. Thus overlap-source coverage is **{int(cov['numerator']):,}/{int(cov['denominator']):,} = {float(cov['value']):.6%}**. Missing values are not set to zero.

The all-valid-row burden share is estimable because the direct-source sum is observed for every valid target row. Its numerator and denominator equal the positive-burden-row sums: the excluded 79 rows add zero to both. The previous audit's phrase “79 other valid-target rows have no observed value among these sources” is contradicted by the current inputs and is not supported by the prior code's mask.

## Suggested author action

The audit recalculation supports the ratio on all 9,631 valid-target rows and shows it equals the positive-burden subset ratio (9,552 rows). The draft's ratio values are numerically reproducible, but its accompanying 9,552/79 coverage sentence is wrong: 9,552 is the positive-burden mask count, and all 9,631 valid rows have observed direct-source values. The author should amend that interpretation, not change the paper during this audit. No claim about indirect category overlap or deterministic full-label reconstruction is supported by this ratio.

Recalculation code: `src/evidence_closure_audit.py::calculate_overlap_reconciliation` and `denominator_audit`. Input file hash is recorded in `reproducibility_manifest.json`.
"""
    (OUT / "denominator_correction_note.md").write_text(text, encoding="utf-8")


def write_report(matrix: pd.DataFrame, denom: pd.DataFrame) -> None:
    counts = matrix.overlap_type.value_counts().to_dict()
    p = denom.set_index("statistic")
    missing_audit = pd.read_csv(OUT / "missingness_semantics_audit.csv")
    source_component_rows = int(missing_audit.source_file.eq(CANON.relative_to(ROOT).as_posix()).sum())
    integration = pd.read_csv(OUT / "integration_reconciliation_recheck.csv")
    alias_recheck = integration[integration.comparison.eq("raw_aliases_to_final")]
    alias_count = len(alias_recheck)
    alias_mismatches = int(pd.to_numeric(alias_recheck.value_mismatches, errors="coerce").fillna(0).sum())
    report = f"""# Final evidence-closure report

## Scope and baseline

This audit began with the repository status captured in `initial_git_status.txt`; substantial modified and untracked project files pre-existed this task. Existing closure outputs and the target, feature, integration, missingness, sequence, and split code were inspected before adding the evidence-specific audit. The frozen target thresholds remain 107, 237, 362, 563 (fit 2001–2010); partitions remain train 2001–2010, validation 2011–2012, locked test 2013–2014. No data, labels, target backup, locked keys, or saved results were written by this audit.

## Issue findings

### 1. Target–predictor overlap — direct confirmed; indirect blocked

The target sums the 11 `BASE_PAPER_GROUPS` with `min_count=1`, then sums group totals with `min_count=1`; any observed negative component invalidates the row. The six grouped predictor formulas are row sums with `min_count=1`. The matrix accounts for all 19 predictor source columns and confirms **9 direct shared IPC components** across five grouped predictors. Confirmed indirect NCRB counting overlaps: **0**. Potential semantic relations: **{counts.get('POSSIBLE_SEMANTIC_OVERLAP', 0)}** mapped predictor-source rows; additional unresolved source rows: **{counts.get('UNRESOLVED', 0)}**. No `NO_IDENTIFIED_OVERLAP` is used as a claim of independence. The direct sources feed the tabular baseline/enhanced feature families and their LSTM/GRU historical sequences; same-year notebook classification is separate.

On the 9,631 valid rows, shared components contribute {float(p.loc['All-valid-target-row burden share','value']):.6%} of summed burden. The prior code's 9,552-row mask excludes 79 valid rows with zero burden and observed zero direct-source contribution. The direct components therefore reconstruct a portion of target burden on all valid rows, not the remaining burden or exact class labels. Confirmed indirect event-count duplication is not established.

The repository relationship review records legal and empirical category relationships, but explicitly says those do not prove NCRB’s event-counting behavior. Official year-specific definitions/crosswalks and aggregation rules for 2001–2014 were not located. This issue remains externally blocked.

### 2. Historical feature availability — structural checks pass; operational origin undefined

The project defines keyed t−1 tabular inputs and strict same-district t−3..t−1 sequence inputs, but does not define a forecast issue date or operational decision point. The official OGD catalog gives catalog publication dates: the 2001–2012 resource was published 17 September 2013; the 2013 resource, which supplies t−1 inputs for the locked 2014 forecasts, was published 16 October 2014; the separate 2014 resource was published 18 August 2016 and is not used as a t−1 predictor for a 2014 target. The catalog describes figures as counts and notes police districts/special units may differ from revenue districts. These are catalog dates, not actual NCRB release dates, retrieval dates, or a particular forecaster’s access date. Each ledger row therefore reports `SCENARIO_UNDEFINED`; a strict date check with a known origin and unknown actual release date returns `UNKNOWN`, never available. Sources: [OGD IPC catalog](https://www.data.gov.in/catalog/district-wise-crimes-under-various-sections-indian-penal-code-ipc-crimes) and [2013 resource](https://www.data.gov.in/resource/district-wise-crime-under-various-sections-indian-penal-code-ipc-crimes-during-2013).

### 3. Provenance, duplicates, and boundaries — partial or externally blocked

The repository contains normalized, cleaned, core, and integrated processed files, but no `data/raw` directory or verified original source download. Processed-file hashes and known transformation code are listed without treating processed files as raw downloads. Read-only re-canonicalization rechecked **{alias_count} integrated IPC aliases** with **{alias_mismatches} value mismatches**. Seven duplicate normalized key groups (14 rows) across five cleaned files were recomputed and compared to the earlier audit. Five have identical offence payloads; two 2010 Railway groups conflict in IPC and women records. Pipeline row IDs are not proven NCRB identifiers; both conflicting groups remain excluded by the existing quality rule, with no aggregation or record selection. The OGD catalog’s police-district/special-unit caveat confirms a distinction from revenue districts but supplies no historical boundary crosswalk. Original data, Railway unit identifiers, district codes, and merger/split histories are required to close these issues.

### 4. Missingness — counts/formulas verified; causes unknown

The audit recomputes **{source_component_rows} IPC source-column/year rows** from the canonical integrated file, plus group and engineered-feature counts. It keeps observed zero, positive, negative, missing, prior-year gap, excluded duplicate, and schema mismatch records separate. Aggregation code preserves `min_count=1`; all-missing group inputs remain missing, partial aggregates are not complete totals, and missing year-over-year inputs remain missing. `CRIME_DIVERSITY_INDEX` treats an individual missing group as zero contribution within its entropy calculation, while an all-groups-missing row remains missing; this is feature-specific behavior, not a blanket dataset fill. No documented reason is assigned to an individual missing source cell. The distinction between structural gap and unknown cause is marked where keyed prior-year evidence permits; missingness cause remains `UNKNOWN_CAUSE` absent original forms or source reporting notes.

### 5. Denominator — corrected reproducibly

The all-valid-row ratio is {p.loc['All-valid-target-row burden share','numerator']:,.0f}/{p.loc['All-valid-target-row burden share','denominator']:,.0f} = {float(p.loc['All-valid-target-row burden share','value']):.6%}, with 9,631/9,631 rows having observed overlap components. The original positive-burden mask gives the same ratio on 9,552 rows because the 79 omitted rows have zero numerator and denominator contribution. The earlier 79-row missing-source interpretation was incorrect. See `denominator_correction_note.md`; the draft remains unchanged.

## Validation and preservation

The evidence-specific regression tests cover all nine direct inputs, availability statuses, denominator masks and sums, missingness/aggregation semantics, duplicate-key preservation, frozen thresholds, chronological partitions, and protected input hashes. Test and audit outcomes are recorded in `test_report.txt`. Models were not retrained. Protected hashes are compared with the initial snapshot and recorded in `reproducibility_manifest.json`; any mismatch fails the generator. The paper and saved experiment outputs are not modified.

## Exact evidence still needed

1. NCRB year-specific district IPC tables/crosswalks and documentation stating whether aggregate and subcategory counts overlap or are mutually exclusive, including applicable years.
2. A declared forecast origin plus actual release/issue records for each annual NCRB source resource; catalog publication and retrieval records must remain distinct.
3. The original downloaded files, stable source identifiers, URL/catalog IDs, retrieval dates and hashes linking them to normalized files.
4. Original Railway unit names/codes and NCRB records for the two conflicting 2010 keys.
5. Official year-specific district/state codes and merger/split boundary tables for 2001–2014.
6. Original source reporting notes or extraction logs identifying why missing IPC component cells occur, including cells in invalid-target rows.
"""
    (OUT / "evidence_closure_audit_report.md").write_text(report, encoding="utf-8")


def write_changed_files() -> None:
    new_outputs = [p.name for p in OUT.iterdir() if p.is_file() and p.name != "changed_files.txt"]
    files = [
        "src/evidence_closure_audit.py — reproducible evidence-closure generator and testable audit functions.",
        "tests/test_final_evidence_closure.py — regression tests for overlap, availability, denominator, missingness, duplicate, split, threshold, and immutability checks.",
    ]
    files.extend(f"outputs/audits/final_evidence_closure/{name} — generated evidence-closure artifact." for name in sorted(new_outputs))
    files.append("outputs/audits/final_evidence_closure/changed_files.txt — this inventory.")
    (OUT / "changed_files.txt").write_text("\n".join(files) + "\n", encoding="utf-8")


def write_manifest(before: dict, after: dict) -> None:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "tensorflow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    outputs = {p.name: sha256(p) for p in OUT.iterdir() if p.is_file() and p.name != "reproducibility_manifest.json"}
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "packages": packages,
        "frozen_thresholds": [107, 237, 362, 563], "threshold_fit_years": [2001, 2010],
        "split": {"train": [2001, 2010], "validation": [2011, 2012], "locked_test": [2013, 2014]},
        "model_retraining": False, "source_or_processed_inputs_modified": False, "paper_modified": False,
        "protected_hashes_before": before, "protected_hashes_after": after,
        "protected_hashes_unchanged": before == after,
        "script_hashes": {"src/evidence_closure_audit.py": sha256(ROOT / "src/evidence_closure_audit.py"),
                          "tests/test_final_evidence_closure.py": sha256(ROOT / "tests/test_final_evidence_closure.py")},
        "output_hashes_sha256": outputs,
        "official_catalog_evidence": {"OGD_catalog": OGD_CATALOG, "2001_2012_catalog_publication": "2013-09-17",
                                      "2013_resource_publication": "2014-10-16", "2014_resource_publication": "2016-08-18",
                                      "actual_release_and_retrieval_dates": "not established"},
        "commands_executed_in_this_closure": ["python src/evidence_closure_audit.py",
                     "python -m unittest tests.test_evidence_closure_audit",
                     "python -m unittest discover -s tests", "python -m compileall -q src ML_Implementation tests"],
        "source_folder_search": {"data_raw_directory_present": (ROOT / "data/raw").exists(),
                                  "original_downloads_verified": False,
                                  "note": "Only processed-stage files were found under data/; their original-download identity is unverified."},
    }
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
