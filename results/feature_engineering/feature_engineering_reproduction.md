# Feature Engineering Reproduction

## Outputs
- derivative_preprocessed.csv: 2015 rows, 48 columns, 46 features excluding Number/PPCs
- derivative_engineered.csv: 2015 rows, 66 columns, 64 features excluding Number/PPCs
- feature_registry.json counts: {'model_a_tree': 53, 'model_b_tree': 64, 'lr_baseline_a': 32, 'lr_baseline_b': 36}

## Key Findings
- PPCs: 454/2015 (22.531%)
- Sex_bin AUROC: 0.620
- Current smoking MI: 0.0108
- Model A total MI: 0.8999
- Model B total MI: 1.0714
- Relative MI gain: 19.1%

## Top AUROC
| feature                            |    auroc |   mutual_info |
|:-----------------------------------|---------:|--------------:|
| PPCs severity                      | 1        |    0.534292   |
| Operation duration                 | 0.686847 |    0.0471032  |
| OpDur_log                          | 0.686847 |    0.0467169  |
| ASA physical status classification | 0.623122 |    0.0356493  |
| Age                                | 0.622448 |    0.0187013  |
| Sex_bin                            | 0.620477 |    0.00163987 |
| Height                             | 0.605874 |    0.0205621  |
| FEV1/FVC                           | 0.60442  |    0.0162818  |
| Intraoperative input               | 0.600914 |    0.0076834  |
| NPAR                               | 0.5985   |    0.00781312 |
| NPAR_log                           | 0.598498 |    0.0067277  |
| ALI                                | 0.590536 |    0.0123797  |
| ALI_log                            | 0.590536 |    0.0127642  |
| LMR                                | 0.587526 |    0          |
| LMR_log                            | 0.587512 |    0.00275781 |
| NLR_log                            | 0.586394 |    0          |
| NLR                                | 0.58639  |    0          |
| FVC%pred                           | 0.585447 |    0.0323753  |
| FEV1%pred                          | 0.578986 |    0.0184977  |
| Neutrophil percentage              | 0.572718 |    0.00617471 |

## Top Mutual Information
| feature                            |    auroc |   mutual_info |
|:-----------------------------------|---------:|--------------:|
| PPCs severity                      | 1        |     0.534292  |
| Operation duration                 | 0.686847 |     0.0471032 |
| OpDur_log                          | 0.686847 |     0.0467169 |
| ASA physical status classification | 0.623122 |     0.0356493 |
| FVC%pred                           | 0.585447 |     0.0323753 |
| Intraoperative blood products      | 0.506428 |     0.0280782 |
| Height                             | 0.605874 |     0.0205621 |
| Coronary heart disease             | 0.51881  |     0.0193029 |
| Single port                        | 0.500646 |     0.0192367 |
| Age                                | 0.622448 |     0.0187013 |
| FEV1%pred                          | 0.578986 |     0.0184977 |
| Glucocorticoids                    | 0.505017 |     0.0166721 |
| FEV1/FVC                           | 0.60442  |     0.0162818 |
| MVV%pred                           | 0.562455 |     0.0154259 |
| PCA pump                           | 0.50681  |     0.014268  |

## Notes
共 9 个衍生指标，除 GNRI 外均取对数。已删除缺失指示变量。Weight 保留并已加入 LR 基线模型。