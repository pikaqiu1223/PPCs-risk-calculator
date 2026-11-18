from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ==========================================
# 1. 路径与配置（自动向上搜索 results/models/locked_model_metadata.json 所在目录）
# ==========================================

def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    env_root = os.environ.get("PPCS_PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    candidates.append(here.parent.parent)
    candidates.append(here.parent)

    for cand in candidates:
        p = cand
        for _ in range(5):
            if (p / "results" / "models" / "locked_model_metadata.json").exists():
                return p
            if p.parent == p:
                break
            p = p.parent
    return here.parent.parent


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = RESULTS_DIR / "models"
FEATURE_REG_PATH = RESULTS_DIR / "feature_selection" / "feature_registry_v3.json"
METADATA_PATH = MODEL_DIR / "locked_model_metadata.json"
INVENTORY_PATH = MODEL_DIR / "locked_model_inventory.csv"

# 全局诊断信息（模型加载失败时记录原因）
_MODEL_LOAD_ERROR: str | None = None

FEATURE_LABELS = {
    "Age": "Age (years)",
    "Sex_bin": "Sex",
    "BMI": "BMI (kg/m²)",
    "Weight": "Weight (kg)",
    "Current smoking": "Current smoking",
    "Malignant tumor history": "Malignant tumor history",
    "Previous pulmonary surgery": "Previous pulmonary surgery",
    "ASA physical status classification": "ASA physical status",
    "Lymphocyte count": "Lymphocyte (×10⁹/L)",
    "Neutrophil count": "Neutrophil (×10⁹/L)",
    "Platelet count": "Platelet (×10⁹/L)",
    "Triglycerides": "Triglycerides (mmol/L)",
    "Hemoglobin": "Hemoglobin (g/dL)",
    "Albumin": "Albumin (g/L)",
    "FVC%pred": "FVC% predicted",
    "FEV1/FVC": "FEV1/FVC",
    "MEF75%pred": "MEF75% predicted",
    "MEF25%pred": "MEF25% predicted",
    "MVV%pred": "MVV% predicted",
    "Surgery_code": "Surgery type",
    "Operation duration": "Operation duration (min)",
    "Intraoperative input": "Intraoperative input (mL)",
    "Glucocorticoids": "Glucocorticoids",
    "SII_log": "Log SII",
    "GNRI": "GNRI",
    "OpDur_log": "Log Operation Duration",
    "Comorbidity_count": "Comorbidity count",
    "Hypertension": "Hypertension",
    "Diabetes": "Diabetes",
    "Coronary heart disease": "Coronary heart disease",
    "Cerebral infarction": "Cerebral infarction",
    "Chronic lung disease": "Chronic lung disease",
}

SURGERY_MAP = {"Lobectomy": 1, "Segmentectomy": 2, "Wedge resection": 3}
SEX_MAP = {"Female": 0, "Male": 1}
YES_NO_MAP = {"No": 0, "Yes": 1}

st.set_page_config(
    page_title="PPCs Risk Calculator (Full)",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px;}
    .risk-panel {
        border: 1px solid #d7dde5;
        border-radius: 10px;
        padding: 20px;
        background: #f8fafc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .risk-value {font-size: 2.8rem; font-weight: 700; color: #0f172a;}
    .risk-label {font-size: 0.9rem; color: #64748b; margin-bottom: 5px;}
    .flag-high {color: #dc2626; font-weight: bold;}
    .flag-low {color: #16a34a; font-weight: bold;}
    div[data-testid="stNumberInput"] > div > input { font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 2. 数据加载与缓存
# ==========================================

@st.cache_resource
def load_registry() -> dict | None:
    if not FEATURE_REG_PATH.exists():
        return None
    with open(FEATURE_REG_PATH, encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_metadata() -> dict | None:
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_defaults() -> dict[str, float]:
    data_path = DATA_DIR / "modeling_cohort_engineered.csv"
    df = pd.DataFrame()
    try:
        if data_path.exists():
            df = pd.read_csv(data_path)
    except Exception:
        df = pd.DataFrame()

    RANGE = {
        "Age": (18, 100, 70),
        "BMI": (10, 60, 24),
        "Weight": (30, 200, 65),
        "Height": (100, 220, 165),
        "Lymphocyte count": (0.1, 20, 2.0),
        "Neutrophil count": (0.1, 50, 4.0),
        "Platelet count": (10, 1000, 200),
        "Triglycerides": (0.1, 20, 1.5),
        "Hemoglobin": (2.0, 20.0, 12.0),
        "Albumin": (10, 60, 40),
        "FVC%pred": (10, 150, 90),
        "FEV1/FVC": (10, 150, 80),
        "MEF75%pred": (0, 300, 80),
        "MEF25%pred": (0, 300, 50),
        "MVV%pred": (0, 300, 80),
        "Operation duration": (10, 600, 84),
        "Intraoperative input": (0, 10000, 1000),
        "Sex_bin": (0, 1, 0),
        "ASA physical status classification": (1, 4, 2),
        "Current smoking": (0, 1, 0),
        "Malignant tumor history": (0, 1, 0),
        "Previous pulmonary surgery": (0, 1, 0),
        "Glucocorticoids": (0, 1, 0),
        "Surgery_code": (1, 3, 1),
    }

    defaults = {}
    for col in RANGE.keys():
        min_val, max_val, fallback = RANGE[col]
        val = fallback
        if col in df.columns and not df[col].isna().all():
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    v = float(df[col].median())
                else:
                    mode_series = df[col].mode()
                    v = float(mode_series.iloc[0]) if not mode_series.empty else fallback
                # 血红蛋白单位自动转换 (g/L -> g/dL)，静默处理
                if col == "Hemoglobin" and v > 20.0:
                    v = v / 10.0
                if min_val <= v <= max_val:
                    val = v
            except Exception:
                val = fallback
        defaults[col] = val

    if "Sex_bin" not in defaults: defaults["Sex_bin"] = 0.0
    if "Height" not in defaults: defaults["Height"] = 165.0
    if "Weight" not in defaults: defaults["Weight"] = 65.0

    return defaults

def _resolve_model_path(model_name: str, info: dict, letter: str) -> Path | None:
    """三级查找模型文件：metadata 记录路径 -> 标准相对路径 -> 目录扫描按 A/B 匹配"""
    models_dir = PROJECT_ROOT / "results" / "models"
    tried: list[Path] = []

    raw = info.get("joblib_path")
    if raw:
        p = Path(str(raw))
        if p.is_absolute():
            tried.append(p)                      # 本地绝对路径（Windows 下有效）
        else:
            tried.append(PROJECT_ROOT / p)       # 相对路径按仓库根拼接

    tried.append(models_dir / f"{model_name}.joblib")
    tried.append(models_dir / f"{model_name.replace('_final', '')}.joblib")

    for c in tried:
        try:
            if c.is_file():
                return c
        except OSError:
            continue

    # 兜底：扫描 models 目录，按 A/B 字母模糊匹配
    if models_dir.is_dir():
        for f in sorted(models_dir.glob("*.joblib")):
            stem = f.stem.upper()
            if letter == "A" and ("_A" in stem or stem.endswith("A")):
                return f
            if letter == "B" and ("_B" in stem or stem.endswith("B")):
                return f
    return None

@st.cache_resource
def get_models(_metadata: dict) -> tuple[dict, dict] | None:
    global _MODEL_LOAD_ERROR
    _MODEL_LOAD_ERROR = None

    best_models = _metadata.get("best_models", {})
    model_a_name = best_models.get("Model_A_final", "LightGBM_A")
    model_b_name = best_models.get("Model_B_final", "LightGBM_B")

    all_models_info = _metadata.get("models", {})
    path_a = _resolve_model_path(model_a_name, all_models_info.get(model_a_name, {}), "A")
    path_b = _resolve_model_path(model_b_name, all_models_info.get(model_b_name, {}), "B")

    if path_a is None or path_b is None:
        _MODEL_LOAD_ERROR = (
            f"Could not locate joblib files. "
            f"Model A name: {model_a_name}, Model B name: {model_b_name}."
        )
        return None

    registry = load_registry()
    if registry is None:
        _MODEL_LOAD_ERROR = "feature_registry_v3.json could not be loaded."
        return None

    features = {
        "A": registry.get("Model_A_final", []),
        "B": registry.get("Model_B_final", []),
    }
    if not features["A"] or not features["B"]:
        _MODEL_LOAD_ERROR = (
            "Feature registry loaded but keys 'Model_A_final'/'Model_B_final' "
            f"are missing or empty. Registry keys found: {list(registry.keys())}"
        )
        return None

    try:
        obj_a = joblib.load(path_a)
        obj_b = joblib.load(path_b)
    except Exception as e:
        _MODEL_LOAD_ERROR = f"joblib.load failed: {e}"
        return None

    def extract(m):
        if isinstance(m, dict):
            return m.get("model", m)
        return m

    return {"A": extract(obj_a), "B": extract(obj_b)}, features

@st.cache_data
def load_thresholds(_metadata: dict) -> tuple[float, float]:
    table_paths = [
        RESULTS_DIR / "external_validation" / "table3_external_threshold_performance.csv",
        PROJECT_ROOT / "manuscript" / "tables" / "table3_external_threshold_performance.csv",
    ]

    th_a, th_b = None, None
    for p in table_paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                col_model = col_algo = col_th = None
                for c in df.columns:
                    if "Model" in c: col_model = c
                    if "Algorithm" in c: col_algo = c
                    if "Threshold" in c: col_th = c

                if col_model and col_th and col_algo:
                    best_a_name = _metadata.get("best_models", {}).get("Model_A_final", "LightGBM_A")
                    best_b_name = _metadata.get("best_models", {}).get("Model_B_final", "LightGBM_B")

                    row_a = df[(df[col_model].str.contains("Model A", case=False, na=False)) & (df[col_algo].str.contains(best_a_name.split("_")[0], case=False, na=False))]
                    row_b = df[(df[col_model].str.contains("Model B", case=False, na=False)) & (df[col_algo].str.contains(best_b_name.split("_")[0], case=False, na=False))]

                    if not row_a.empty:
                        val = str(row_a.iloc[0][col_th])
                        th_a = float(val.split("=")[-1].strip()) if "=" in val else float(val)
                    if not row_b.empty:
                        val = str(row_b.iloc[0][col_th])
                        th_b = float(val.split("=")[-1].strip()) if "=" in val else float(val)
            except Exception as e:
                print(f"Error loading table 3: {e}")

    if th_a is None or th_b is None:
        return 0.38, 0.37
    return th_a, th_b

# ==========================================
# 3. 计算逻辑
# ==========================================

def calculate_ideal_weight(height_cm: float, sex_bin: int) -> float:
    height_in = height_cm / 2.54
    base_height_in = 60.0
    if sex_bin == 1:
        base_w, factor = 48.0, 2.7
    else:
        base_w, factor = 45.5, 2.2
    if height_in < base_height_in:
        return base_w
    return base_w + factor * (height_in - base_height_in)

def compute_derived_features(user_input: dict, defaults: dict) -> dict:
    res = {}
    comorbidities = [
        "Hypertension", "Diabetes", "Coronary heart disease",
        "Cerebral infarction", "Chronic lung disease",
        "Current smoking", "Malignant tumor history", "Previous pulmonary surgery"
    ]
    res["Comorbidity_count"] = float(sum(1 for c in comorbidities if user_input.get(c, 0) == 1))

    platelets = user_input.get("Platelet count", defaults.get("Platelet count", 200.0))
    neu = user_input.get("Neutrophil count", defaults.get("Neutrophil count", 4.0))
    lym = user_input.get("Lymphocyte count", defaults.get("Lymphocyte count", 2.0))
    sii = (platelets * neu) / lym if lym > 0 else 0
    res["SII_log"] = float(np.log1p(sii))

    alb = user_input.get("Albumin", defaults.get("Albumin", 40.0))
    w_act = user_input.get("Weight", defaults.get("Weight", 65.0))
    h = user_input.get("Height", defaults.get("Height", 165.0))
    sex = user_input.get("Sex_bin", 0)
    w_ideal = calculate_ideal_weight(h, sex)
    ratio = w_act / w_ideal if w_ideal > 0 else 1.0
    res["GNRI"] = min(max(1.489 * alb + 41.7 * ratio, 60.0), 150.0)

    op_dur = user_input.get("Operation duration", defaults.get("Operation duration", 84.0))
    res["OpDur_log"] = float(np.log1p(op_dur))
    return res

def build_prediction_vector(user_input: dict, derived: dict, features: list[str], defaults: dict) -> pd.DataFrame:
    row = {}
    for f in features:
        if f in user_input:
            row[f] = user_input[f]
        elif f in derived:
            row[f] = derived[f]
        else:
            row[f] = defaults.get(f, 0.0)
    return pd.DataFrame([row], columns=features)

# ==========================================
# 4. 界面渲染
# ==========================================

def _show_pyplot(fig):
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        st.pyplot(fig, use_container_width=True)

def render_input_section(defaults: dict, key_prefix: str, include_intraop: bool) -> dict:
    st.markdown("### 📋 Patient Inputs")
    inputs = {}

    c1, c2, c3 = st.columns(3)
    with c1:
        inputs["Age"] = st.number_input("Age (years)", min_value=18, max_value=100, value=int(defaults.get("Age", 70)), key=f"{key_prefix}_age")
    with c2:
        inputs["Sex_bin"] = SEX_MAP[st.selectbox("Sex", list(SEX_MAP.keys()), index=0, key=f"{key_prefix}_sex")]
    with c3:
        inputs["BMI"] = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=float(defaults.get("BMI", 24.0)), key=f"{key_prefix}_bmi")

    c1, c2 = st.columns(2)
    with c1:
        inputs["Height"] = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=float(defaults.get("Height", 165.0)), key=f"{key_prefix}_height")
    with c2:
        inputs["Weight"] = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=float(defaults.get("Weight", 65.0)), key=f"{key_prefix}_weight")

    st.markdown("#### Comorbidities")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        inputs["Hypertension"] = YES_NO_MAP[st.selectbox("Hypertension", list(YES_NO_MAP.keys()), index=0, key=f"{key_prefix}_htn")]
        inputs["Diabetes"] = YES_NO_MAP[st.selectbox("Diabetes", list(YES_NO_MAP.keys()), index=0, key=f"{key_prefix}_dm")]
    with c2:
        inputs["Coronary heart disease"] = YES_NO_MAP[st.selectbox("CAD", list(YES_NO_MAP.keys()), index=0, key=f"{key_prefix}_cad")]
        inputs["Cerebral infarction"] = YES_NO_MAP[st.selectbox("Cerebral Infarction", list(YES_NO_MAP.keys()), index=0, key=f"{key_prefix}_stroke")]
    with c3:
        inputs["Chronic lung disease"] = YES_NO_MAP[st.selectbox("Chronic Lung Disease", list(YES_NO_MAP.keys()), index=0, key=f"{key_prefix}_copd")]
        inputs["Current smoking"] = YES_NO_MAP[st.selectbox("Current Smoker", list(YES_NO_MAP.keys()), index=0, key=f"{key_prefix}_smoke")]
    with c4:
        inputs["Malignant tumor history"] = YES_NO_MAP[st.selectbox("Malignancy", list(YES_NO_MAP.keys()), index=0, key=f"{key_prefix}_cancer")]
        inputs["Previous pulmonary surgery"] = YES_NO_MAP[st.selectbox("Prev. Pulmonary Surgery", list(YES_NO_MAP.keys()), index=0, key=f"{key_prefix}_prev_surg")]

    st.markdown("#### Functional & Surgical Status")
    c1, c2, c3 = st.columns(3)
    with c1:
        inputs["ASA physical status classification"] = int(st.selectbox("ASA Physical Status", [1, 2, 3, 4], index=1, key=f"{key_prefix}_asa"))
    with c2:
        inputs["Surgery_code"] = SURGERY_MAP[st.selectbox("Surgery Type", list(SURGERY_MAP.keys()), index=0, key=f"{key_prefix}_surg")]
    with c3:
        inputs["FVC%pred"] = st.number_input("FVC% predicted", min_value=10.0, max_value=150.0, value=float(defaults.get("FVC%pred", 90.0)), key=f"{key_prefix}_fvc")

    st.markdown("#### Laboratory Tests")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        inputs["Neutrophil count"] = st.number_input("Neutrophil (×10⁹/L)", min_value=0.0, max_value=50.0, value=float(defaults.get("Neutrophil count", 4.0)), key=f"{key_prefix}_neu")
    with c2:
        inputs["Lymphocyte count"] = st.number_input("Lymphocyte (×10⁹/L)", min_value=0.0, max_value=20.0, value=float(defaults.get("Lymphocyte count", 2.0)), key=f"{key_prefix}_lymp")
    with c3:
        inputs["Platelet count"] = st.number_input("Platelet (×10⁹/L)", min_value=0.0, max_value=1000.0, value=float(defaults.get("Platelet count", 200.0)), key=f"{key_prefix}_plt")
    with c4:
        inputs["Albumin"] = st.number_input("Albumin (g/L)", min_value=10.0, max_value=60.0, value=float(defaults.get("Albumin", 40.0)), key=f"{key_prefix}_alb")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        inputs["Triglycerides"] = st.number_input("Triglycerides (mmol/L)", min_value=0.0, max_value=20.0, value=float(defaults.get("Triglycerides", 1.5)), key=f"{key_prefix}_tg")
    with c2:
        inputs["Hemoglobin"] = st.number_input("Hemoglobin (g/dL)", min_value=2.0, max_value=20.0, value=float(defaults.get("Hemoglobin", 12.0)), key=f"{key_prefix}_hgb")
    with c3:
        inputs["FEV1/FVC"] = st.number_input("FEV1/FVC", min_value=10.0, max_value=150.0, value=float(defaults.get("FEV1/FVC", 80.0)), key=f"{key_prefix}_fev1")
    with c4:
        inputs["MEF75%pred"] = st.number_input("MEF75% predicted", min_value=0.0, max_value=300.0, value=float(defaults.get("MEF75%pred", 80.0)), key=f"{key_prefix}_mef75")

    c1, c2 = st.columns(2)
    with c1:
        inputs["MEF25%pred"] = st.number_input("MEF25% predicted", min_value=0.0, max_value=300.0, value=float(defaults.get("MEF25%pred", 50.0)), key=f"{key_prefix}_mef25")
    with c2:
        inputs["MVV%pred"] = st.number_input("MVV% predicted", min_value=0.0, max_value=300.0, value=float(defaults.get("MVV%pred", 80.0)), key=f"{key_prefix}_mvv")

    if include_intraop:
        st.markdown("#### 🛠️ Intraoperative Factors")
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs["Operation duration"] = st.number_input("Op Duration (min)", min_value=10, max_value=600, value=int(defaults.get("Operation duration", 84)), key=f"{key_prefix}_opdur")
        with c2:
            inputs["Intraoperative input"] = st.number_input("Fluid Input (mL)", min_value=0, max_value=10000, value=int(defaults.get("Intraoperative input", 1000)), key=f"{key_prefix}_input")
        with c3:
            inputs["Glucocorticoids"] = YES_NO_MAP[st.selectbox("Glucocorticoids", list(YES_NO_MAP.keys()), index=0, key=f"{key_prefix}_gluco")]

    return inputs

def render_results(risk: float, threshold: float, model_name: str, model_obj, x_df, features):
    st.markdown(f"### 🎯 Prediction Results ({model_name})")

    col1, col2, col3 = st.columns([1, 1, 1.5])
    with col1:
        st.markdown(f"<div class='risk-panel'><div class='risk-label'>PPCs Risk</div><div class='risk-value'>{risk:.1%}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='risk-panel'><div class='risk-label'>Optimal Threshold</div><div class='risk-value'>{threshold:.1%}</div></div>", unsafe_allow_html=True)
    with col3:
        color = "flag-high" if risk >= threshold else "flag-low"
        st.markdown(f"<div class='risk-panel'><div class='risk-label'>Status</div><div class='risk-value {color}'>{'>= Threshold' if risk >= threshold else '< Threshold'}</div></div>", unsafe_allow_html=True)

    st.markdown("#### 🔍 Feature Contribution (SHAP)")
    try:
        data_path = DATA_DIR / "modeling_cohort_engineered.csv"
        bg_data = x_df
        if data_path.exists():
            bg_df = pd.read_csv(data_path)
            if all(c in bg_df.columns for c in features) and len(bg_df) > 0:
                bg_data = bg_df[features].sample(min(100, len(bg_df)), random_state=42)

        explainer = shap.TreeExplainer(model_obj, bg_data)
        shap_values = explainer.shap_values(x_df)

        if isinstance(shap_values, list): shap_values = shap_values[1]
        if shap_values.ndim > 2: shap_values = shap_values[:, :, -1]
        if shap_values.ndim == 1: shap_values = shap_values.reshape(1, -1)

        vals = shap_values[0]
        feat_names = [FEATURE_LABELS.get(f, f) for f in features]

        top_indices = np.argsort(np.abs(vals))[::-1][:8][::-1]
        plot_feats = [feat_names[i] for i in top_indices]
        plot_vals = vals[top_indices]
        colors = ["#dc2626" if v > 0 else "#2563eb" for v in plot_vals]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(plot_feats, plot_vals, color=colors, height=0.66)
        ax.axvline(0, color="#475569", lw=0.8)
        ax.set_xlabel("SHAP Value (Impact on Risk)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        _show_pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Could not generate SHAP plot: {e}")

# ==========================================
# 5. 主程序
# ==========================================

def main():
    st.title("PPCs Risk Calculator (Enhanced)")
    st.caption("Including accurate calculation of SII and GNRI based on full lab parameters.")

    # === 免责声明 ===
    st.warning("""
    ⚠️ RESEARCH PROTOTYPE & DISCLAIMER

    For Research Use Only: This online calculator is a research prototype intended for scientific reference only. The calculated risk does not constitute medical advice and must not replace comprehensive clinical judgment.

    Local Calibration Required: The predicted probabilities are based on a model trained on our derivation cohort. Before applying this tool in a new institution, local intercept calibration (calibration-in-the-large) using local data is mandatory to account for potential population differences.

    Exploratory Clinical Utility: The proposed decision thresholds and clinical pathways are exploratory and require prospective validation. Do not use this tool as the sole trigger for clinical interventions (e.g., ICU admission).
    """)

    metadata = load_metadata()
    if metadata is None:
        st.error("Model metadata not found. App cannot start.")
        st.info(f"Checked path: {METADATA_PATH}")
        st.stop()

    loaded = get_models(metadata)
    if loaded is None:
        st.error("Model files or feature registry not found. App cannot start.")
        st.info(f"Project root in use: `{PROJECT_ROOT}`")

        models_dir = PROJECT_ROOT / "results" / "models"
        if models_dir.is_dir():
            files = sorted(f.name for f in models_dir.iterdir() if f.is_file())
            st.info("Files actually present in results/models/: " + (", ".join(files) if files else "(empty)"))
        else:
            st.info(f"Directory does not exist on the server: `{models_dir}`")

        st.info(f"feature_registry_v3.json exists: {FEATURE_REG_PATH.exists()}")

        st.info(f"best_models from metadata: {metadata.get('best_models', {})}")
        if _MODEL_LOAD_ERROR:
            st.error("Diagnosis detail: " + _MODEL_LOAD_ERROR)
        st.stop()

    models, features = loaded
    defaults = load_defaults()
    thresholds = load_thresholds(metadata)

    best_a_name = metadata.get("best_models", {}).get("Model_A_final", "LightGBM_A")
    best_b_name = metadata.get("best_models", {}).get("Model_B_final", "LightGBM_B")
    threshold_a, threshold_b = thresholds

    tab1, tab2, tab3 = st.tabs(["Model A (Preop)", "Model B (Intraop)", "Comparison"])

    with tab1:
        left, right = st.columns([1, 1])
        with left:
            inputs_a = render_input_section(defaults, "A", include_intraop=False)
            derived_a = compute_derived_features(inputs_a, defaults)
        with right:
            x_a = build_prediction_vector(inputs_a, derived_a, features["A"], defaults)
            risk_a = models["A"].predict_proba(x_a)[0, 1]
            render_results(risk_a, threshold_a, best_a_name, models["A"], x_a, features["A"])

    with tab2:
        left, right = st.columns([1, 1])
        with left:
            inputs_b = render_input_section(defaults, "B", include_intraop=True)
            derived_b = compute_derived_features(inputs_b, defaults)
        with right:
            x_b = build_prediction_vector(inputs_b, derived_b, features["B"], defaults)
            risk_b = models["B"].predict_proba(x_b)[0, 1]
            render_results(risk_b, threshold_b, best_b_name, models["B"], x_b, features["B"])

    with tab3:
        st.info("This section allows you to compare the risk trajectory from Preop (Model A) to Intraop (Model B) using the same patient data.")
        inputs_cmp = render_input_section(defaults, "Cmp", include_intraop=True)
        derived_cmp = compute_derived_features(inputs_cmp, defaults)

        x_for_a = build_prediction_vector(inputs_cmp, derived_cmp, features["A"], defaults)
        x_for_b = build_prediction_vector(inputs_cmp, derived_cmp, features["B"], defaults)

        risk_preop = models["A"].predict_proba(x_for_a)[0, 1]
        risk_intraop = models["B"].predict_proba(x_for_b)[0, 1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Preoperative Risk (Model A)", f"{risk_preop:.1%}")
        col2.metric("Intraop Updated Risk (Model B)", f"{risk_intraop:.1%}", delta=f"{risk_intraop - risk_preop:+.1%}")
        col3.metric("Change", f"{risk_intraop - risk_preop:+.1%}")

        st.markdown("### Detailed Explanation (Model B)")
        render_results(risk_intraop, threshold_b, best_b_name, models["B"], x_for_b, features["B"])

if __name__ == "__main__":
    main()
