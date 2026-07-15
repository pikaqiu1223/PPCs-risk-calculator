# Model Tuning Report

## CV-AUC Comparison

| feature_set   | model              |   n_features |   mean_auc |   sd_auc |   n_trials |
|:--------------|:-------------------|-------------:|-----------:|---------:|-----------:|
| Model_A_final | LightGBM           |           19 |     0.7395 |   0.0287 |         75 |
| Model_A_final | XGBoost            |           19 |     0.7383 |   0.0311 |         75 |
| Model_A_final | CatBoost           |           19 |     0.7343 |   0.0337 |         75 |
| Model_A_final | LogisticRegression |           18 |     0.7143 |   0.0106 |          0 |
| Model_B_final | LightGBM           |           27 |     0.775  |   0.0245 |         75 |
| Model_B_final | XGBoost            |           27 |     0.7748 |   0.0259 |         75 |
| Model_B_final | CatBoost           |           27 |     0.7737 |   0.0252 |         75 |
| Model_B_final | LogisticRegression |           26 |     0.7644 |   0.0189 |          0 |

## Warnings

_No LR-over-tree warning triggered._

## Best Tuned Parameters

### Model_A_final

**LightGBM**



```json

{
  "n_estimators": 250,
  "learning_rate": 0.00920467834975137,
  "num_leaves": 14,
  "max_depth": 7,
  "min_child_samples": 15,
  "subsample": 0.8585039203394088,
  "subsample_freq": 2,
  "colsample_bytree": 0.502345175182721,
  "reg_alpha": 0.00018347720713648335,
  "reg_lambda": 0.00029392367402301755,
  "min_split_gain": 0.5847668648999942,
  "max_bin": 203,
  "class_weight": null
}

```

**XGBoost**



```json

{
  "n_estimators": 273,
  "learning_rate": 0.008064004236114246,
  "max_depth": 6,
  "min_child_weight": 16.937172645747335,
  "subsample": 0.866506516640362,
  "colsample_bytree": 0.45212365426120527,
  "colsample_bylevel": 0.7619057826873772,
  "gamma": 1.3243162469081515,
  "reg_alpha": 0.3993453328009045,
  "reg_lambda": 0.01286031326546596,
  "max_bin": 105,
  "scale_pos_weight_multiplier": 1.728079400904977
}

```

**CatBoost**



```json

{
  "iterations": 472,
  "learning_rate": 0.00984007022636947,
  "depth": 5,
  "l2_leaf_reg": 62.847874057133104,
  "random_strength": 0.3693949775677883,
  "bagging_temperature": 3.3924679689703017,
  "border_count": 103,
  "rsm": 0.7710931481134974,
  "scale_pos_weight_multiplier": 1.0915410644385375
}

```

**LogisticRegression**



```json

{
  "C": 0.1,
  "max_iter": 1000,
  "class_weight": "balanced",
  "solver": "liblinear"
}

```

### Model_B_final

**LightGBM**



```json

{
  "n_estimators": 720,
  "learning_rate": 0.007721772086664055,
  "num_leaves": 63,
  "max_depth": 5,
  "min_child_samples": 56,
  "subsample": 0.6486442284174269,
  "subsample_freq": 2,
  "colsample_bytree": 0.4733546577771604,
  "reg_alpha": 0.014973208787835058,
  "reg_lambda": 3.6091759627741915,
  "min_split_gain": 0.02371955508619647,
  "max_bin": 129,
  "class_weight": null
}

```

**XGBoost**



```json

{
  "n_estimators": 318,
  "learning_rate": 0.012591372878875592,
  "max_depth": 7,
  "min_child_weight": 20.91890526023464,
  "subsample": 0.5570663781299804,
  "colsample_bytree": 0.7955589378354138,
  "colsample_bylevel": 0.5040498478337094,
  "gamma": 3.643548914190526,
  "reg_alpha": 0.01617304239912193,
  "reg_lambda": 0.0001898682640821184,
  "max_bin": 94,
  "scale_pos_weight_multiplier": 1.0068425890534285
}

```

**CatBoost**



```json

{
  "iterations": 1159,
  "learning_rate": 0.011165851077236442,
  "depth": 4,
  "l2_leaf_reg": 57.95505100739204,
  "random_strength": 3.5032478152566098,
  "bagging_temperature": 0.05733502860556666,
  "border_count": 186,
  "rsm": 0.8611870700799263,
  "scale_pos_weight_multiplier": 0.514575112878351
}

```

**LogisticRegression**



```json

{
  "C": 0.1,
  "max_iter": 1000,
  "class_weight": "balanced",
  "solver": "liblinear"
}

```

## Methods Text

Hyperparameter tuning was performed exclusively within the derivation cohort using Optuna with a fixed random seed of 42. LightGBM, XGBoost, and CatBoost were tuned separately for Model A-final and Model B-final using the same 5-fold cross-validation splits stratified by hospital-outcome label. In each fold, median imputation was fitted on the training fold only and applied to the validation fold. Logistic regression with standardization was evaluated as a fixed-parameter comparator and was not tuned against external validation data.
