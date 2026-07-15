# v3.1 Audit Addendum (LightGBM-Centered)

## LightGBM vs Best Other Tree Model (XGBoost / CatBoost)

| feature_set   |   n_features |   lightgbm_auc |   lightgbm_sd | best_other_tree_model   |   best_other_tree_auc |   best_other_tree_sd |   lightgbm_minus_other_auc | warning_other_ge_lightgbm   |
|:--------------|-------------:|---------------:|--------------:|:------------------------|----------------------:|---------------------:|---------------------------:|:----------------------------|
| Model_A_final |           19 |         0.7395 |        0.0287 | XGBoost                 |                0.7383 |               0.0311 |                     0.0012 | False                       |
| Model_B_final |           27 |         0.775  |        0.0245 | XGBoost                 |                0.7748 |               0.0259 |                     0.0002 | False                       |

## LightGBM vs Logistic Regression (Reference)

| feature_set   |   lightgbm_auc |   lr_auc |   lightgbm_minus_lr_auc | lr_within_0.005_of_lightgbm   |
|:--------------|---------------:|---------:|------------------------:|:------------------------------|
| Model_A_final |         0.7395 |   0.7143 |                  0.0252 | False                         |
| Model_B_final |         0.775  |   0.7644 |                  0.0106 | False                         |

## scale_pos_weight_multiplier Confirmation

scale_pos_weight_multiplier is an Optuna-only tuning variable. It is removed from the parameter dictionary with params_local.pop('scale_pos_weight_multiplier', 1.0), multiplied by the empirical negative-to-positive class ratio, and passed to XGBoost/CatBoost using their standard scale_pos_weight constructor argument.

Code references:

- model_tuning.py:123-130 for XGBoost
- model_tuning.py:140-148 for CatBoost
- model_tuning.py:199 and model_tuning.py:213 for the Optuna search variable

## Note

All comparisons are performed using the same 5-fold cross-validation splits (stratified by hospital-outcome). LightGBM is the primary model; other models are shown for completeness.
