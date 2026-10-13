# IPC source relationship review

## What is supported

The nine direct shared IPC source variables remain direct overlap. Their identity is a code/data fact; it does not show the proportion of individual cases counted twice.

The official Indian Penal Code provides three relevant legal relationships: section 390 says robbery can consist of theft when the statutory conditions are met; section 391 defines dacoity as five or more people conjointly committing or attempting robbery; and section 396 is dacoity with murder. Sections 299–300 define culpable homicide and murder and distinguish culpable homicide not amounting to murder. These establish semantic relationships among the legal definitions, not the NCRB table's case-counting or mutually-exclusive tabulation practice.

The Government of India's Open Government Data catalog identifies the district-wise IPC data as sourced from Crime in India 2014 and lists the crime heads. The catalog page does not document whether counts across parent/subhead columns are mutually exclusive or whether one case can appear in more than one heading. Original source copies and a relevant NCRB table note/codebook are not present in the project repository.

## Conclusions

- `IPC_THEFT`, `IPC_ROBBERY`, and `IPC_DACOITY` have a legally supported conceptual relationship. The amount of duplicate counting in the project data remains unresolved.
- `IPC_MURDER` and target source `IPC_DACOITY_WITH_MURDER` also have a legally supported relationship under section 396. Whether the data records these in both counts is unresolved.
- Local parent-child comparisons support investigation of rape, kidnapping, riot, dacoity, and theft subheads. Their observed numerical relationships do not establish a formal NCRB hierarchy or duplicated case accounting.
- Consequently, direct overlap is established, some indirect semantic relationships are supported, and event-level/data-accounting overlap is still unresolved. No deterministic target reconstruction or absence-of-overlap claim is made.
- The random-split notebook remains observed-year classification. Forecast results use only historical t−1 or t−3…t−1 inputs structurally; release timing remains unverified.

## Sources

- India Code, Indian Penal Code, sections 299–300 and 390–396: https://www.indiacode.nic.in/bitstream/123456789/15289/1/ipc_act.pdf
- Government of India Open Government Data, district-wise IPC crime catalog (attributes data to NCRB Crime in India 2014): https://www.data.gov.in/catalog/district-wise-crimes-under-various-sections-indian-penal-code-ipc-crimes
- Repository empirical source comparison: `outputs/tables/crime_parent_child_validation.csv`.
- Project direct source mapping: `outputs/tables/target_predictor_source_overlap.csv` and `outputs/tables/target_source_column_mapping.csv`.
