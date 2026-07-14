# Feature Selection Report

Input data file: `modeling_cohort_engineered.csv`

## CV-AUC Summary

| Stage                                 |   Feature count |   CV-AUC |
|:--------------------------------------|----------------:|---------:|
| Stage 1 survivors / Stage 2 no-change |              42 |   0.7443 |
| Stage 3 final Model B                 |              27 |   0.7543 |
| Stage 3 final Model A                 |              19 |   0.7046 |
| Validation check: LR-B                |              26 |   0.7644 |


**WARNING:** Logistic regression using Model_LR_B matched or exceeded LightGBM using Model_B_final in 5-fold CV. This affects the primary hypothesis that ML should outperform traditional models and should be reported transparently.

## Stage 1 - Collinearity Filter

| Dropped feature       | Kept feature     |   Spearman abs r |   Dropped AUROC |   Kept AUROC | Reason                            |
|:----------------------|:-----------------|-----------------:|----------------:|-------------:|:----------------------------------|
| Operation duration    | OpDur_log        |            1     |           0.687 |        0.687 | AUROC tie; clinical priority rule |
| ALI                   | ALI_log          |            1     |           0.591 |        0.591 | AUROC tie; clinical priority rule |
| SII                   | SII_log          |            1     |           0.568 |        0.568 | AUROC tie; clinical priority rule |
| NPAR                  | NPAR_log         |            1     |           0.599 |        0.598 | AUROC tie; clinical priority rule |
| dNLR                  | dNLR_log         |            1     |           0.572 |        0.572 | AUROC tie; clinical priority rule |
| NLR                   | NLR_log          |            1     |           0.586 |        0.586 | AUROC tie; clinical priority rule |
| AAPR                  | AAPR_log         |            1     |           0.527 |        0.527 | AUROC tie; clinical priority rule |
| PLR                   | PLR_log          |            1     |           0.53  |        0.53  | AUROC tie; clinical priority rule |
| LMR                   | LMR_log          |            1     |           0.588 |        0.588 | AUROC tie; clinical priority rule |
| Neutrophil percentage | dNLR_log         |            0.999 |           0.573 |        0.572 | AUROC tie; clinical priority rule |
| dNLR_log              | NLR_log          |            0.966 |           0.572 |        0.586 | lower single-variable AUROC       |
| Alkaline phosphatase  | AAPR_log         |            0.948 |           0.511 |        0.527 | lower single-variable AUROC       |
| NLR_log               | ALI_log          |            0.934 |           0.586 |        0.591 | lower single-variable AUROC       |
| MEF50%pred            | MMEF%pred        |            0.924 |           0.538 |        0.542 | lower single-variable AUROC       |
| ALI_log               | NPAR_log         |            0.905 |           0.591 |        0.598 | lower single-variable AUROC       |
| WBC                   | Neutrophil count |            0.891 |           0.554 |        0.571 | lower single-variable AUROC       |
| MMEF%pred             | MEF25%pred       |            0.855 |           0.542 |        0.562 | lower single-variable AUROC       |
| FEV1%pred             | FVC%pred         |            0.844 |           0.579 |        0.585 | lower single-variable AUROC       |

Stage 1 survivors (42):

`Age`, `BMI`, `Hypertension`, `Diabetes`, `Coronary heart disease`, `Cerebral infarction`, `Chronic lung disease`, `Current smoking`, `Malignant tumor history`, `Previous pulmonary surgery`, `ASA physical status classification`, `Single port`, `Intraoperative input`, `Dexmedetomidine`, `NSAIDs`, `Glucocorticoids`, `Intraoperative blood products`, `Nerve block`, `PCA pump`, `Neutrophil count`, `Lymphocyte count`, `Monocyte count`, `Platelet count`, `Albumin`, `Triglycerides`, `Hemoglobin`, `FVC%pred`, `FEV1/FVC`, `MEF25%pred`, `MVV%pred`, `Sex_bin`, `Surgery_code`, `Anesthesia_balanced`, `NMB_reversal`, `PLR_log`, `LMR_log`, `SII_log`, `NPAR_log`, `AAPR_log`, `GNRI`, `OpDur_log`, `Comorbidity_count`

## Stage 2 - Mutual Information Screen

| feature                            |   mutual_information | low_information_candidate   |
|:-----------------------------------|---------------------:|:----------------------------|
| OpDur_log                          |              0.05177 | No                          |
| Age                                |              0.03112 | No                          |
| FVC%pred                           |              0.02937 | No                          |
| ASA physical status classification |              0.02199 | No                          |
| MVV%pred                           |              0.0211  | No                          |
| Sex_bin                            |              0.02058 | No                          |
| FEV1/FVC                           |              0.01568 | No                          |
| Hemoglobin                         |              0.01492 | No                          |
| Intraoperative input               |              0.01319 | No                          |
| AAPR_log                           |              0.01046 | No                          |
| SII_log                            |              0.00964 | No                          |
| NPAR_log                           |              0.00786 | No                          |
| Neutrophil count                   |              0.00698 | No                          |
| Albumin                            |              0.00689 | No                          |
| Current smoking                    |              0.00636 | No                          |
| Comorbidity_count                  |              0.00601 | No                          |
| Surgery_code                       |              0.00535 | No                          |
| BMI                                |              0.00328 | No                          |
| PLR_log                            |              0.00268 | Yes                         |
| Monocyte count                     |              0.00238 | Yes                         |
| Intraoperative blood products      |              0.00212 | Yes                         |
| Chronic lung disease               |              0.00202 | Yes                         |
| Cerebral infarction                |              0.00179 | Yes                         |
| Coronary heart disease             |              0.00176 | Yes                         |
| GNRI                               |              0.0012  | Yes                         |
| LMR_log                            |              0.00094 | Yes                         |
| PCA pump                           |              0.00092 | Yes                         |
| Hypertension                       |              0.00054 | Yes                         |
| Nerve block                        |              0.00048 | Yes                         |
| Malignant tumor history            |              0.00026 | Yes                         |
| Anesthesia_balanced                |              6e-05   | Yes                         |
| NMB_reversal                       |              6e-05   | Yes                         |
| Glucocorticoids                    |              4e-05   | Yes                         |
| Previous pulmonary surgery         |              2e-05   | Yes                         |
| Dexmedetomidine                    |              0       | Yes                         |
| NSAIDs                             |              0       | Yes                         |
| Single port                        |              0       | Yes                         |
| Diabetes                           |              0       | Yes                         |
| Triglycerides                      |              0       | Yes                         |
| Platelet count                     |              0       | Yes                         |
| Lymphocyte count                   |              0       | Yes                         |
| MEF25%pred                         |              0       | Yes                         |

## Stage 3 - Backward Elimination by CV-AUC Contribution

|   Try order | Feature                            | Low MI   |   Gain importance |   Baseline CV-AUC |   Reduced CV-AUC |   Delta AUC | Decision             |   Feature count after |
|------------:|:-----------------------------------|:---------|------------------:|------------------:|-----------------:|------------:|:---------------------|----------------------:|
|           1 | Intraoperative blood products      | Yes      |              0    |            0.7443 |           0.7443 |      0      | dropped              |                    41 |
|           2 | PCA pump                           | Yes      |              0.06 |            0.7443 |           0.7451 |     -0.0008 | dropped              |                    40 |
|           3 | Malignant tumor history            | Yes      |              3.67 |            0.7451 |           0.7534 |     -0.0083 | dropped              |                    39 |
|           4 | Previous pulmonary surgery         | Yes      |             20.43 |            0.7534 |           0.7459 |      0.0075 | kept_delta_too_large |                    39 |
|           5 | Coronary heart disease             | Yes      |             22.62 |            0.7534 |           0.7461 |      0.0074 | kept_delta_too_large |                    39 |
|           6 | Cerebral infarction                | Yes      |             31.86 |            0.7534 |           0.7451 |      0.0083 | kept_delta_too_large |                    39 |
|           7 | Single port                        | Yes      |             32.67 |            0.7534 |           0.7502 |      0.0032 | kept_delta_too_large |                    39 |
|           8 | Anesthesia_balanced                | Yes      |             37.32 |            0.7534 |           0.7471 |      0.0063 | kept_delta_too_large |                    39 |
|           9 | Diabetes                           | Yes      |             42.56 |            0.7534 |           0.7541 |     -0.0006 | dropped              |                    38 |
|          10 | Chronic lung disease               | Yes      |             43.05 |            0.7541 |           0.7499 |      0.0042 | kept_delta_too_large |                    38 |
|          11 | NMB_reversal                       | Yes      |             46.63 |            0.7541 |           0.7457 |      0.0083 | kept_delta_too_large |                    38 |
|          12 | Dexmedetomidine                    | Yes      |             47.37 |            0.7541 |           0.7443 |      0.0097 | kept_delta_too_large |                    38 |
|          13 | Nerve block                        | Yes      |             48.86 |            0.7541 |           0.7451 |      0.0089 | kept_delta_too_large |                    38 |
|          14 | Hypertension                       | Yes      |             57.18 |            0.7541 |           0.7463 |      0.0078 | kept_delta_too_large |                    38 |
|          15 | NSAIDs                             | Yes      |             93.51 |            0.7541 |           0.7415 |      0.0126 | kept_delta_too_large |                    38 |
|          16 | Glucocorticoids                    | Yes      |            127.83 |            0.7541 |           0.7475 |      0.0066 | kept_delta_too_large |                    38 |
|          17 | Lymphocyte count                   | Yes      |            629.59 |            0.7541 |           0.7507 |      0.0033 | kept_delta_too_large |                    38 |
|          18 | Monocyte count                     | Yes      |            643.62 |            0.7541 |           0.7525 |      0.0015 | dropped              |                    37 |
|          19 | GNRI                               | Yes      |            704    |            0.7525 |           0.7513 |      0.0012 | dropped              |                    36 |
|          20 | Platelet count                     | Yes      |            725.45 |            0.7513 |           0.7517 |     -0.0004 | dropped              |                    35 |
|          21 | LMR_log                            | Yes      |            847.41 |            0.7517 |           0.7511 |      0.0006 | dropped              |                    34 |
|          22 | PLR_log                            | Yes      |            911.18 |            0.7511 |           0.7457 |      0.0054 | kept_delta_too_large |                    34 |
|          23 | Triglycerides                      | Yes      |           1044.68 |            0.7511 |           0.7498 |      0.0013 | dropped              |                    33 |
|          24 | MEF25%pred                         | Yes      |           1097.1  |            0.7498 |           0.7435 |      0.0063 | kept_delta_too_large |                    33 |
|          25 | Current smoking                    | No       |            112.56 |            0.7498 |           0.7536 |     -0.0038 | dropped              |                    32 |
|          26 | Surgery_code                       | No       |            132.18 |            0.7536 |                  |             | kept_forced_include  |                    32 |
|          27 | Comorbidity_count                  | No       |            191.96 |            0.7536 |           0.7514 |      0.0022 | dropped              |                    31 |
|          28 | Intraoperative input               | No       |            272.66 |            0.7514 |                  |             | kept_forced_include  |                    31 |
|          29 | Sex_bin                            | No       |            602.52 |            0.7514 |                  |             | kept_forced_include  |                    31 |
|          30 | SII_log                            | No       |            653.3  |            0.7514 |           0.746  |      0.0054 | kept_delta_too_large |                    31 |
|          31 | Hemoglobin                         | No       |            693.07 |            0.7514 |           0.7488 |      0.0026 | dropped              |                    30 |
|          32 | Albumin                            | No       |            694.66 |            0.7488 |           0.7515 |     -0.0027 | dropped              |                    29 |
|          33 | NPAR_log                           | No       |            800.34 |            0.7515 |           0.7561 |     -0.0046 | dropped              |                    28 |
|          34 | Neutrophil count                   | No       |            803.24 |            0.7561 |           0.7543 |      0.0018 | dropped              |                    27 |
|          35 | BMI                                | No       |            826.38 |            0.7543 |           0.7476 |      0.0067 | kept_delta_too_large |                    27 |
|          36 | ASA physical status classification | No       |            947.25 |            0.7543 |           0.7375 |      0.0168 | kept_delta_too_large |                    27 |
|          37 | MVV%pred                           | No       |            967.38 |            0.7543 |                  |             | kept_forced_include  |                    27 |
|          38 | Age                                | No       |           1001.19 |            0.7543 |                  |             | kept_forced_include  |                    27 |
|          39 | AAPR_log                           | No       |           1077.29 |            0.7543 |           0.7503 |      0.004  | kept_delta_too_large |                    27 |
|          40 | FEV1/FVC                           | No       |           1301.59 |            0.7543 |                  |             | kept_forced_include  |                    27 |
|          41 | FVC%pred                           | No       |           1321.16 |            0.7543 |           0.7491 |      0.0051 | kept_delta_too_large |                    27 |
|          42 | OpDur_log                          | No       |           2527.23 |            0.7543 |           0.7155 |      0.0388 | kept_delta_too_large |                    27 |

## Final Feature Sets

Model_B_final (27 features):

`Age`, `BMI`, `Hypertension`, `Coronary heart disease`, `Cerebral infarction`, `Chronic lung disease`, `Previous pulmonary surgery`, `ASA physical status classification`, `Single port`, `Intraoperative input`, `Dexmedetomidine`, `NSAIDs`, `Glucocorticoids`, `Nerve block`, `Lymphocyte count`, `FVC%pred`, `FEV1/FVC`, `MEF25%pred`, `MVV%pred`, `Sex_bin`, `Surgery_code`, `Anesthesia_balanced`, `NMB_reversal`, `PLR_log`, `SII_log`, `AAPR_log`, `OpDur_log`

Model_A_final (19 features):

`Age`, `BMI`, `Hypertension`, `Coronary heart disease`, `Cerebral infarction`, `Chronic lung disease`, `Previous pulmonary surgery`, `ASA physical status classification`, `Single port`, `Lymphocyte count`, `FVC%pred`, `FEV1/FVC`, `MEF25%pred`, `MVV%pred`, `Sex_bin`, `Surgery_code`, `PLR_log`, `SII_log`, `AAPR_log`

Model_LR_B (26 features):

`Age`, `BMI`, `Hypertension`, `Coronary heart disease`, `Cerebral infarction`, `Chronic lung disease`, `Previous pulmonary surgery`, `ASA physical status classification`, `Single port`, `Intraoperative input`, `Dexmedetomidine`, `NSAIDs`, `Glucocorticoids`, `Nerve block`, `Lymphocyte count`, `FVC%pred`, `FEV1/FVC`, `MEF25%pred`, `Sex_bin`, `Surgery_code`, `Anesthesia_balanced`, `NMB_reversal`, `PLR_log`, `SII_log`, `AAPR_log`, `OpDur_log`

## Methods Text

Candidate predictors were first prespecified according to prediction time point and clinical availability. Redundant predictors were removed within the derivation cohort using pairwise Spearman correlation, with highly correlated pairs (absolute r > 0.80) resolved by single-variable AUROC and prespecified clinical priority rules. Mutual information was then computed for all Stage 1 survivors to flag low-information predictors without automatically excluding them. Final parsimonious predictors were selected by backward elimination using 5-fold cross-validated LightGBM AUROC stratified by hospital-outcome label; a feature was removed only when its exclusion reduced mean CV-AUC by less than 0.003. All preprocessing inside cross-validation was fitted on the training fold only, and random seeds were fixed at 42.
