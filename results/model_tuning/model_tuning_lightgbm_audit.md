# LightGBM-Centric Model Comparison Report

## LightGBM vs Best Other Tree Model (XGBoost / CatBoost)

| feature_set   |   lightgbm_auc |   lightgbm_sd | best_other_model   |   best_other_auc |   best_other_sd |   lightgbm_minus_other | other_better   |
|:--------------|---------------:|--------------:|:-------------------|-----------------:|----------------:|-----------------------:|:---------------|
| Model_A_final |         0.7395 |        0.0287 | XGBoost            |           0.7383 |          0.0311 |                 0.0012 | False          |
| Model_B_final |         0.775  |        0.0245 | XGBoost            |           0.7748 |          0.0259 |                 0.0002 | False          |

## LightGBM vs Logistic Regression (Reference)

| feature_set   |   lightgbm_auc |   lr_auc |   lightgbm_minus_lr | lr_within_0.005   |
|:--------------|---------------:|---------:|--------------------:|:------------------|
| Model_A_final |         0.7395 |   0.7143 |              0.0252 | False             |
| Model_B_final |         0.775  |   0.7644 |              0.0106 | False             |

## Note

All comparisons are based on the same 5-fold cross-validation splits (stratified by hospital-outcome). LightGBM is the primary model; other models are shown for completeness.
