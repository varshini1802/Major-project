# Corrected v5 multi-seed robustness results

Five predeclared seeds (11, 23, 37, 53, 71) completed. Each run reused the corrected-v4 training procedure, frozen thresholds, 2001–2010 train / 2011–2012 validation / 2013–2014 test split, and identical locked 671-key test cohort. Each paired GRU/LSTM baseline and enhanced model used validation macro-F1 for epoch selection; logistic C selection was also validation-only. No test result selected a model.

## Paired differences by seed

Positive values favor enhanced features. These are fixed-test descriptive differences, not causal effects.

```text
 seed              model                      metric  baseline  enhanced  enhanced_minus_baseline
   11                GRU               test_accuracy  0.719821  0.727273                 0.007452
   11                GRU               test_macro_f1  0.706313  0.716970                 0.010656
   11                GRU            test_weighted_f1  0.727961  0.734113                 0.006152
   11                GRU           macro_ovr_roc_auc  0.937530  0.936334                -0.001196
   11                GRU macro_ovr_average_precision  0.720966  0.724907                 0.003942
   11               LSTM               test_accuracy  0.715350  0.719821                 0.004471
   11               LSTM               test_macro_f1  0.702067  0.707629                 0.005562
   11               LSTM            test_weighted_f1  0.723609  0.726286                 0.002677
   11               LSTM           macro_ovr_roc_auc  0.938601  0.935782                -0.002819
   11               LSTM macro_ovr_average_precision  0.722154  0.721489                -0.000665
   11 LogisticRegression               test_accuracy  0.695976  0.697466                 0.001490
   11 LogisticRegression               test_macro_f1  0.676082  0.676243                 0.000161
   11 LogisticRegression            test_weighted_f1  0.701935  0.702644                 0.000708
   11 LogisticRegression           macro_ovr_roc_auc  0.932690  0.931229                -0.001461
   11 LogisticRegression macro_ovr_average_precision  0.709531  0.705922                -0.003609
   23                GRU               test_accuracy  0.716841  0.712370                -0.004471
   23                GRU               test_macro_f1  0.703204  0.697866                -0.005337
   23                GRU            test_weighted_f1  0.723622  0.717437                -0.006186
   23                GRU           macro_ovr_roc_auc  0.936129  0.933721                -0.002408
   23                GRU macro_ovr_average_precision  0.720041  0.712412                -0.007629
   23               LSTM               test_accuracy  0.715350  0.724292                 0.008942
   23               LSTM               test_macro_f1  0.700064  0.705390                 0.005326
   23               LSTM            test_weighted_f1  0.721889  0.728708                 0.006819
   23               LSTM           macro_ovr_roc_auc  0.936548  0.934213                -0.002335
   23               LSTM macro_ovr_average_precision  0.716956  0.716833                -0.000123
   23 LogisticRegression               test_accuracy  0.695976  0.697466                 0.001490
   23 LogisticRegression               test_macro_f1  0.676082  0.676243                 0.000161
   23 LogisticRegression            test_weighted_f1  0.701935  0.702644                 0.000708
   23 LogisticRegression           macro_ovr_roc_auc  0.932690  0.931229                -0.001461
   23 LogisticRegression macro_ovr_average_precision  0.709531  0.705922                -0.003609
   37                GRU               test_accuracy  0.724292  0.731744                 0.007452
   37                GRU               test_macro_f1  0.703106  0.710679                 0.007573
   37                GRU            test_weighted_f1  0.727744  0.735437                 0.007693
   37                GRU           macro_ovr_roc_auc  0.938663  0.938557                -0.000106
   37                GRU macro_ovr_average_precision  0.724535  0.728029                 0.003494
   37               LSTM               test_accuracy  0.715350  0.727273                 0.011923
   37               LSTM               test_macro_f1  0.693093  0.703620                 0.010527
   37               LSTM            test_weighted_f1  0.719622  0.730534                 0.010913
   37               LSTM           macro_ovr_roc_auc  0.936798  0.937765                 0.000966
   37               LSTM macro_ovr_average_precision  0.718470  0.722783                 0.004313
   37 LogisticRegression               test_accuracy  0.695976  0.697466                 0.001490
   37 LogisticRegression               test_macro_f1  0.676082  0.676243                 0.000161
   37 LogisticRegression            test_weighted_f1  0.701935  0.702644                 0.000708
   37 LogisticRegression           macro_ovr_roc_auc  0.932690  0.931229                -0.001461
   37 LogisticRegression macro_ovr_average_precision  0.709531  0.705922                -0.003609
   53                GRU               test_accuracy  0.721311  0.731744                 0.010432
   53                GRU               test_macro_f1  0.699577  0.712611                 0.013035
   53                GRU            test_weighted_f1  0.725224  0.735638                 0.010414
   53                GRU           macro_ovr_roc_auc  0.938990  0.937610                -0.001380
   53                GRU macro_ovr_average_precision  0.723898  0.722894                -0.001004
   53               LSTM               test_accuracy  0.719821  0.736215                 0.016393
   53               LSTM               test_macro_f1  0.698303  0.720280                 0.021978
   53               LSTM            test_weighted_f1  0.724149  0.740219                 0.016070
   53               LSTM           macro_ovr_roc_auc  0.937780  0.938486                 0.000706
   53               LSTM macro_ovr_average_precision  0.720853  0.733753                 0.012900
   53 LogisticRegression               test_accuracy  0.695976  0.697466                 0.001490
   53 LogisticRegression               test_macro_f1  0.676082  0.676243                 0.000161
   53 LogisticRegression            test_weighted_f1  0.701935  0.702644                 0.000708
   53 LogisticRegression           macro_ovr_roc_auc  0.932690  0.931229                -0.001461
   53 LogisticRegression macro_ovr_average_precision  0.709531  0.705922                -0.003609
   71                GRU               test_accuracy  0.713860  0.721311                 0.007452
   71                GRU               test_macro_f1  0.690176  0.703495                 0.013319
   71                GRU            test_weighted_f1  0.717601  0.726372                 0.008771
   71                GRU           macro_ovr_roc_auc  0.934259  0.934381                 0.000122
   71                GRU macro_ovr_average_precision  0.711007  0.712975                 0.001968
   71               LSTM               test_accuracy  0.713860  0.728763                 0.014903
   71               LSTM               test_macro_f1  0.692454  0.711457                 0.019003
   71               LSTM            test_weighted_f1  0.719008  0.733770                 0.014763
   71               LSTM           macro_ovr_roc_auc  0.934907  0.935560                 0.000653
   71               LSTM macro_ovr_average_precision  0.712108  0.719743                 0.007635
   71 LogisticRegression               test_accuracy  0.695976  0.697466                 0.001490
   71 LogisticRegression               test_macro_f1  0.676082  0.676243                 0.000161
   71 LogisticRegression            test_weighted_f1  0.701935  0.702644                 0.000708
   71 LogisticRegression           macro_ovr_roc_auc  0.932690  0.931229                -0.001461
   71 LogisticRegression macro_ovr_average_precision  0.709531  0.705922                -0.003609
```

## Mean and standard deviation of paired differences

```text
             model                      metric  seeds  mean_paired_difference  standard_deviation           direction_counts                                                                                                       interval
               GRU macro_ovr_average_precision      5                0.000154            0.004762 higher=3; equal=0; lower=2 Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort.
               GRU           macro_ovr_roc_auc      5               -0.000993            0.001028 higher=1; equal=0; lower=4 Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort.
               GRU               test_accuracy      5                0.005663            0.005810 higher=4; equal=0; lower=1 Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort.
               GRU               test_macro_f1      5                0.007849            0.007724 higher=4; equal=0; lower=1 Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort.
               GRU            test_weighted_f1      5                0.005369            0.006644 higher=4; equal=0; lower=1 Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort.
              LSTM macro_ovr_average_precision      5                0.004812            0.005656 higher=3; equal=0; lower=2 Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort.
              LSTM           macro_ovr_roc_auc      5               -0.000566            0.001848 higher=3; equal=0; lower=2 Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort.
              LSTM               test_accuracy      5                0.011326            0.004783 higher=5; equal=0; lower=0 Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort.
              LSTM               test_macro_f1      5                0.012479            0.007675 higher=5; equal=0; lower=0 Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort.
              LSTM            test_weighted_f1      5                0.010248            0.005563 higher=5; equal=0; lower=0 Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort.
LogisticRegression macro_ovr_average_precision      5               -0.003609            0.000000 higher=0; equal=0; lower=5 Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort.
LogisticRegression           macro_ovr_roc_auc      5               -0.001461            0.000000 higher=0; equal=0; lower=5 Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort.
LogisticRegression               test_accuracy      5                0.001490            0.000000 higher=5; equal=0; lower=0 Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort.
LogisticRegression               test_macro_f1      5                0.000161            0.000000 higher=5; equal=0; lower=0 Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort.
LogisticRegression            test_weighted_f1      5                0.000708            0.000000 higher=5; equal=0; lower=0 Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort.
```

## Stability interpretation

GRU enhanced macro-F1 was higher in 4/5 seeds (mean paired difference +0.00785, seed SD 0.00772), while GRU macro ROC-AUC was lower in 4/5 (mean −0.00099) and AP direction was mixed (3/5 higher). LSTM enhanced macro-F1 was higher in all five seeds (mean +0.01248, seed SD 0.00768), and accuracy and weighted-F1 were higher in all five. LSTM ROC-AUC direction was mixed (3/5 higher), and AP was higher in 3/5. This supports seed-stable LSTM macro-F1 improvement on this locked cohort but does not establish a universal gain across model families or metrics. Logistic regression differences are exactly repeatable here because its fixed procedure is deterministic for this data/configuration.

No confidence interval or p-value is reported. Seeds vary initialization/training randomness; they do not resample districts or years. Test rows from a district and adjacent years can be dependent, so ordinary row-wise intervals would not address that dependence. District-cluster or temporal-block inference was not attempted.

The preserved seed-42 v4 ablation remains separate. The one-feature-at-a-time analyses were also rerun in the distinct v5 five-seed ablation directory (50/50 fits); see its ablation report and integrity audit. Saved v4 outputs remain unchanged.
