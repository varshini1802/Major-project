# Target–predictor overlap resolution

The actual corrected temporal feature contracts use keyed `t-1` tabular values and strict `t-3..t-1` recurrent sequences. The direct nine shared IPC source components occur in historical rows, not target-year `t` inputs. This is historical dependence and does not establish contemporaneous target leakage.

The non-temporal random-split notebook uses same-year crime-group inputs for a target built from same-year IPC counts; this is contemporaneous target-component overlap and is valid only as observed-year classification. It cannot support future-forecast claims.

The prior audit's direct component share is 3,228,843 / 4,215,620 = 76.592364%. The denominator reconciliation is independently recomputed in `denominator_reconciliation.csv`; 9,552 positive-burden rows and 9,631 valid-target rows are distinct masks, with 79 observed zero-burden rows omitted from the former. This measures shared component count, not exact label reconstruction.

Indirect parent/subcategory or cross-domain counting relationships remain unresolved because authoritative year-specific NCRB definitions and aggregation rules are absent. No independence claim is made. Operational availability also remains unresolved because actual release and retrieval timestamps are absent.
