# Stage 3 elimination log

| feature                            |   try_order |   delta_auc | decision             |
|:-----------------------------------|------------:|------------:|:---------------------|
| Intraoperative blood products      |           1 |    0        | dropped              |
| PCA pump                           |           2 |   -0.000831 | dropped              |
| Malignant tumor history            |           3 |   -0.008336 | dropped              |
| Previous pulmonary surgery         |           4 |    0.007489 | kept_delta_too_large |
| Coronary heart disease             |           5 |    0.007363 | kept_delta_too_large |
| Cerebral infarction                |           6 |    0.008333 | kept_delta_too_large |
| Single port                        |           7 |    0.003247 | kept_delta_too_large |
| Anesthesia_balanced                |           8 |    0.006345 | kept_delta_too_large |
| Diabetes                           |           9 |   -0.000639 | dropped              |
| Chronic lung disease               |          10 |    0.004199 | kept_delta_too_large |
| NMB_reversal                       |          11 |    0.00833  | kept_delta_too_large |
| Dexmedetomidine                    |          12 |    0.009723 | kept_delta_too_large |
| Nerve block                        |          13 |    0.008911 | kept_delta_too_large |
| Hypertension                       |          14 |    0.007759 | kept_delta_too_large |
| NSAIDs                             |          15 |    0.012601 | kept_delta_too_large |
| Glucocorticoids                    |          16 |    0.006599 | kept_delta_too_large |
| Lymphocyte count                   |          17 |    0.003349 | kept_delta_too_large |
| Monocyte count                     |          18 |    0.001531 | dropped              |
| GNRI                               |          19 |    0.001242 | dropped              |
| Platelet count                     |          20 |   -0.000387 | dropped              |
| LMR_log                            |          21 |    0.000572 | dropped              |
| PLR_log                            |          22 |    0.005424 | kept_delta_too_large |
| Triglycerides                      |          23 |    0.001252 | dropped              |
| MEF25%pred                         |          24 |    0.006318 | kept_delta_too_large |
| Current smoking                    |          25 |   -0.003764 | dropped              |
| Surgery_code                       |          26 |  nan        | kept_forced_include  |
| Comorbidity_count                  |          27 |    0.002169 | dropped              |
| Intraoperative input               |          28 |  nan        | kept_forced_include  |
| Sex_bin                            |          29 |  nan        | kept_forced_include  |
| SII_log                            |          30 |    0.005396 | kept_delta_too_large |
| Hemoglobin                         |          31 |    0.00264  | dropped              |
| Albumin                            |          32 |   -0.002696 | dropped              |
| NPAR_log                           |          33 |   -0.004605 | dropped              |
| Neutrophil count                   |          34 |    0.001836 | dropped              |
| BMI                                |          35 |    0.00666  | kept_delta_too_large |
| ASA physical status classification |          36 |    0.016816 | kept_delta_too_large |
| MVV%pred                           |          37 |  nan        | kept_forced_include  |
| Age                                |          38 |  nan        | kept_forced_include  |
| AAPR_log                           |          39 |    0.003993 | kept_delta_too_large |
| FEV1/FVC                           |          40 |  nan        | kept_forced_include  |
| FVC%pred                           |          41 |    0.005131 | kept_delta_too_large |
| OpDur_log                          |          42 |    0.038763 | kept_delta_too_large |
