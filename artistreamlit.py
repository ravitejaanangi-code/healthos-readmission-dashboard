# app.py — HealthOS: 30-Day Readmission Risk Dashboard (with LIME)
# --------------------------------------------------------------------------------
# Tabs:
#   1) Predict by Patient (cards show High-Risk Encounters & Total Encounters + LIME local explanation)
#   2) Manual Entry (clinical ranges & hints)
#   3) Insights (no metric cards)
#   4) High-Risk Patients
#   5) Feature Importance
#
# Supports loading artifacts from default dirs or via sidebar uploads.
# Predicts with the active dataset; header cards can optionally use a full dataset.

from sklearn.inspection import permutation_importance
from streamlit_extras.metric_cards import style_metric_cards
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import warnings
import os
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Readmission Risk Dashboard", layout="wide")

# ---------------- Config ----------------
DECISION_THRESHOLD = 0.30  # fixed classification threshold

# Optional: point to a full CSV for header cards (else will use active dataset)
# e.g., r"C:\Users\Tharun\OneDrive\Documents\mrpstreamlit\articrafts123\final_ready_dataset_complete.csv"
FULL_METRICS_CSV = r""

# We try these automatically for artifacts (no folder picker UI)
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DIRS = [BASE_DIR]


# Numeric fields for Manual Entry
NUM_FIELDS = [
    "AGE", "BMI", "Hemoglobin",
    "Systolic_BP", "Diastolic_BP",
    "Heart_Rate", "Respiratory_Rate",
    "LOS"
]
# Clinical ranges (for sliders & feedback)
CLINICAL = {
    "AGE":              {"min": 0,   "max": 100, "note": "Typical adult range 18–90"},
    "BMI":              {"min": 10,  "max": 60,  "normal_low": 18.5, "normal_high": 24.9, "note": "Normal 18.5–24.9"},
    "Hemoglobin_male":  {"min": 5,   "max": 22,  "normal_low": 13.8, "normal_high": 17.2, "unit": "g/dL"},
    "Hemoglobin_female": {"min": 5,   "max": 22,  "normal_low": 12.1, "normal_high": 15.1, "unit": "g/dL"},
    "Systolic_BP":      {"min": 80,  "max": 220, "normal_low": 90,   "normal_high": 120,  "note": "≥130 = high"},
    "Diastolic_BP":     {"min": 40,  "max": 130, "normal_low": 60,   "normal_high": 80,   "note": "≥80 = high"},
    "Heart_Rate":       {"min": 40,  "max": 180, "normal_low": 60,   "normal_high": 100,  "unit": "bpm"},
    "Respiratory_Rate": {"min": 8,   "max": 40,  "normal_low": 12,   "normal_high": 20,   "unit": "/min"},
    "LOS":              {"min": 0,   "max": 60,  "note": "Typical 0–30"}
}

sns.set_theme(style="whitegrid", rc={
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})

st.markdown("""
<style>
    .patient-card {
        background: linear-gradient(to right, #f8fbff, #e6f0ff);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        font-size: 16px;
    }
    .card-line { margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# ====================== ARTIFACT LOADING ======================


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def load_from_dir(art_dir: Path):
    """Return (mode, model, preproc, features, data) if found in art_dir."""
    mode = model = preproc = features = data = None
    pipe = art_dir / "model_pipeline.joblib"
    feats_csv = art_dir / "features.csv"

    if _exists(pipe):
        model = joblib.load(pipe)
        if _exists(feats_csv):
            try:
                df_feats = pd.read_csv(feats_csv)
                features = (df_feats.iloc[:, 0] if df_feats.shape[1] == 1
                            else df_feats["features"]).tolist()
            except Exception:
                features = None
        mode = "pipeline"

    mod = art_dir / "xgb_tuned_model.pkl"
    prep = art_dir / "xgb_tuned_preprocessor.pkl"
    feats2 = art_dir / "xgb_features.csv"
    if model is None and _exists(mod) and _exists(prep):
        model = joblib.load(mod)
        preproc = joblib.load(prep)
        if _exists(feats2):
            features = pd.read_csv(feats2, header=None).iloc[:, 0].tolist()
            features = [f for f in features if f != "0"]
        mode = "separate"

    # 🔹 Try several possible dataset filenames in this artifact directory
    for fname in ["readmission_test_data.csv", "test_dataset.csv", "dataset.csv"]:
        candidate = art_dir / fname
        if _exists(candidate):
            data = pd.read_csv(candidate)
            break

    return mode, model, preproc, features, data


def try_default_dirs():
    for d in DEFAULT_DIRS:
        mode, model, preproc, features, data = load_from_dir(d)
        if model is not None:
            return mode, model, preproc, features, data
    return None, None, None, None, None


def load_from_uploads():
    """Sidebar uploads (models & optional datasets)."""
    st.sidebar.markdown("### 📤 Upload artifacts")
    up_model = st.sidebar.file_uploader(
        "Model (pipeline .joblib OR model .pkl)", type=["joblib", "pkl"])
    up_pre = st.sidebar.file_uploader(
        "Preprocessor (.pkl) — only for separate model", type=["pkl"])
    up_feats = st.sidebar.file_uploader("Features CSV", type=["csv"])
    up_data = st.sidebar.file_uploader("Dataset CSV (optional)", type=["csv"])
    up_full = st.sidebar.file_uploader(
        "Full dataset for metrics (optional)", type=["csv"])

    mode = model = preproc = features = data = full = None
    if up_model is not None:
        try:
            model = joblib.load(BytesIO(up_model.read()))
            mode = "pipeline" if (hasattr(model, "named_steps") or hasattr(
                model, "steps")) else "separate"
        except Exception as e:
            st.sidebar.error(f"Model load error: {e}")
            return None, None, None, None, None, None

    if up_pre is not None:
        try:
            preproc = joblib.load(BytesIO(up_pre.read()))
        except Exception as e:
            st.sidebar.error(f"Preprocessor load error: {e}")
            return None, None, None, None, None, None

    if up_feats is not None:
        try:
            features = pd.read_csv(up_feats, header=None).iloc[:, 0].tolist()
            features = [f for f in features if f != "0"]
        except Exception as e:
            st.sidebar.error(f"Features CSV load error: {e}")
            return None, None, None, None, None, None

    if up_data is not None:
        try:
            data = pd.read_csv(up_data)
        except Exception as e:
            st.sidebar.error(f"Dataset CSV load error: {e}")
            return None, None, None, None, None, None

    if up_full is not None:
        try:
            full = pd.read_csv(up_full)
        except Exception as e:
            st.sidebar.error(f"Full dataset load error: {e}")
            full = None

    return mode, model, preproc, features, data, full


# Try default dirs (no folder UI)
mode, model, preprocessor, features, data = try_default_dirs()

# Upload fallback
full_for_metrics = None
if model is None:
    st.sidebar.info("Artifacts not found on disk. Upload them below.")
    mode, model, preprocessor, features, data, full_for_metrics = load_from_uploads()

if model is None:
    st.error("No model loaded. Provide a pipeline (.joblib) OR a separate model (.pkl) + preprocessor + features.csv.")
    st.stop()

if mode is None:
    mode = "pipeline" if (hasattr(model, "named_steps")
                          or hasattr(model, "steps")) else "separate"


def infer_features_if_missing(model, preproc, fallback_cols=None):
    if preproc is not None:
        try:
            return list(preproc.feature_names_in_)
        except Exception:
            pass
    try:
        prep = model.named_steps.get("prep", None)
        if prep is not None:
            try:
                return list(prep.feature_names_in_)
            except Exception:
                pass
    except Exception:
        pass
    return list(fallback_cols) if fallback_cols is not None else None


if features is None:
    features = infer_features_if_missing(model, preprocessor, fallback_cols=(
        data.columns if isinstance(data, pd.DataFrame) else None))
if features is None:
    st.error(
        "Could not infer feature list. Please provide xgb_features.csv or features.csv.")
    st.stop()

if not isinstance(data, pd.DataFrame):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Dataset")
    up_data_only = st.sidebar.file_uploader(
        "Dataset CSV", type=["csv"], key="data_only")
    if up_data_only is None:
        st.info("Upload a dataset CSV to continue.")
        st.stop()
    data = pd.read_csv(up_data_only)

# ====================== HELPERS ======================


def robust_patient_count(df: pd.DataFrame) -> int:
    if "PATIENT" not in df.columns:
        return len(df)
    s = df["PATIENT"].astype(str).str.strip().replace(
        {"": np.nan, "nan": np.nan, "None": np.nan})
    return s.nunique(dropna=True)


def get_proba(df_feat: pd.DataFrame) -> np.ndarray:
    if mode == "pipeline":
        return model.predict_proba(df_feat[features])[:, 1]
    else:
        Xt = preprocessor.transform(df_feat[features])
        return model.predict_proba(Xt)[:, 1]


def ensure_scores(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    if "Risk_Score" not in df.columns:
        df["Risk_Score"] = get_proba(df)
    df["Risk_Label"] = np.where(
        df["Risk_Score"] >= thr, "High Risk", "Low Risk")
    return df


def num(value, nd=2):
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return "—"


def hemo_norm_range(gender: str):
    g = str(gender).lower()
    if g.startswith("f"):
        return CLINICAL["Hemoglobin_female"]["normal_low"], CLINICAL["Hemoglobin_female"]["normal_high"]
    if g.startswith("m"):
        return CLINICAL["Hemoglobin_male"]["normal_low"], CLINICAL["Hemoglobin_male"]["normal_high"]
    return 12.0, 17.5


def _safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

# LIME helpers


def _lime_background(df: pd.DataFrame, cols):
    """Create a NaN-free background matrix for LIME with simple imputations."""
    X = df[cols].copy()
    for c in cols:
        if c not in X.columns:
            X[c] = np.nan
    for c in cols:
        if pd.api.types.is_numeric_dtype(X[c]):
            med = X[c].median()
            X[c] = X[c].fillna(med)
        else:
            mode = X[c].mode(dropna=True)
            if len(mode) == 0:
                X[c] = X[c].fillna("Unknown")
            else:
                X[c] = X[c].fillna(mode.iloc[0])
    # sample to keep explainer light
    if len(X) > 600:
        X = X.sample(600, random_state=42)
    return X


def _lime_categorical_info(df: pd.DataFrame, cols):
    """Return (categorical_feature_indices, categorical_names_dict) for LIME."""
    cat_idx = []
    cat_names = {}
    for i, c in enumerate(cols):
        if not pd.api.types.is_numeric_dtype(df[c]):
            cat_idx.append(i)
            # limit categories to avoid huge labels
            uniques = list(pd.Series(df[c].dropna().astype(str).unique())[:50])
            cat_names[i] = uniques
    return cat_idx, cat_names

# ====================== APP BODY ======================


st.title("🏥 HealthOS: 30-Day Readmission Risk Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Predict by Patient",
    "📝 Manual Entry",
    "📊 Insights",
    "📋 High-Risk Patients",
    "💡 Feature Importance"
])

# ------------------ TAB 1: Predict by Patient (with LIME) ------------------
with tab1:
    st.subheader("🔍 Predict by Patient")

    # --- METRIC CARDS: High-risk encounters & total encounters (rows) ---
    metrics_df = None
    if FULL_METRICS_CSV and os.path.exists(FULL_METRICS_CSV):
        metrics_df = _safe_read_csv(FULL_METRICS_CSV)
    if metrics_df is None and isinstance(full_for_metrics, pd.DataFrame):
        metrics_df = full_for_metrics
    if metrics_df is None:
        metrics_df = data.copy()

    metrics_df = ensure_scores(metrics_df, DECISION_THRESHOLD)

    total_patients = len(metrics_df)  # rows (encounters)
    highrisk_patients = int((metrics_df["Risk_Label"] == "High Risk").sum())
    readmit_rate = metrics_df["READMISSION_30"].mean(
    ) * 100 if "READMISSION_30" in metrics_df.columns else np.nan
    avg_los = metrics_df["LOS"].mean(
    ) if "LOS" in metrics_df.columns else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 High-Risk Patients", f"{highrisk_patients:,}")
    c2.metric("🏥 Total Patients", f"{total_patients:,}")
    c3.metric("📉 Readmission Rate",
              f"{readmit_rate:.2f}%" if pd.notna(readmit_rate) else "—")
    c4.metric("🛌 Avg LOS", f"{avg_los:.1f} days" if pd.notna(avg_los) else "—")
    style_metric_cards()

    # Patient selector (uses active dataset `data`)
    if "PATIENT" in data.columns and data["PATIENT"].notna().any():
        pat_id = st.selectbox(
            "Choose PATIENT",
            data["PATIENT"].dropna().astype(str).str.strip().unique()
        )
        subset = data[data["PATIENT"].astype(str).str.strip() == pat_id].copy()

        # pick latest encounter if START exists
        if "START" in subset.columns:
            subset["START_DT"] = pd.to_datetime(
                subset["START"], errors="coerce", utc=True)
            chosen = (subset.dropna(subset=["START_DT"]).sort_values("START_DT").iloc[-1]
                      if subset["START_DT"].notna().any() else subset.iloc[-1])
        else:
            chosen = subset.iloc[-1]

        st.markdown("### 🧬 Patient Clinical Summary")
        st.markdown(f"""
        <div class='patient-card'>
            <div class='card-line'><strong>PATIENT:</strong> {pat_id}</div>
            <div class='card-line'><strong>Age:</strong> {chosen.get('AGE', '—')}</div>
            <div class='card-line'><strong>Gender:</strong> {chosen.get('GENDER', '—')}</div>
            <div class='card-line'><strong>Race:</strong> {chosen.get('RACE', '—')}</div>
            <div class='card-line'><strong>Latest Condition:</strong> {chosen.get('DESCRIPTION', '—')}</div>
            <div class='card-line'><strong>Latest Reason:</strong> {chosen.get('REASONDESCRIPTION', '—')}</div>
            <div class='card-line'><strong>BMI:</strong> {num(chosen.get('BMI', np.nan), 1)}</div>
            <div class='card-line'><strong>Length of Stay:</strong> {int(chosen.get('LOS', 0)) if pd.notna(chosen.get('LOS', np.nan)) else '—'} days</div>
            <div class='card-line'><strong>Smoking:</strong> {chosen.get('Smoking_Status', '—')}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🧠 Predict Readmission Risk", key="predict_patient_latest"):
            single_df = pd.DataFrame([chosen[features]])
            prob = float(get_proba(single_df)[0])
            label = "High Risk" if prob >= DECISION_THRESHOLD else "Low Risk"

            st.markdown(f"""
            <div style='background-color:#fff3cd; padding:15px; border-radius:10px;'>
                <strong>🔮 Predicted Probability:</strong>
                <span style='color:#0d6efd; font-size:18px;'>{prob:.2f}</span><br>
                {('<span style="color:red; font-weight:bold;">⚠️ High Risk — Consider early intervention.</span>'
                  if label == 'High Risk'
                  else '<span style="color:green; font-weight:bold;">✅ Low Risk — Patient likely stable.</span>')}
            </div>
            """, unsafe_allow_html=True)
            st.caption(
                f"Decision threshold is fixed at **{DECISION_THRESHOLD:.2f}**. "
                f"Scores ≥ {DECISION_THRESHOLD:.2f} are labeled **High Risk (1)**; otherwise **Low Risk (0)**."
            )
            if "REASONDESCRIPTION" in subset.columns:
                st.info(
                    f"**Reason (REASONDESCRIPTION):** {chosen.get('REASONDESCRIPTION', '—')}")
            elif "DESCRIPTION" in subset.columns:
                st.info(f"**Description:** {chosen.get('DESCRIPTION', '—')}")

# ---------------- LIME Local Explanation (robust with categorical encoding) ----------------
    try:
        from lime.lime_tabular import LimeTabularExplainer

        # 1) Choose the same background the app uses
        bg = data.copy()

        # 2) Identify categorical features among "features"
        cat_cols = [
            c for c in features if c in bg.columns and not pd.api.types.is_numeric_dtype(bg[c])]
        num_cols = [c for c in features if c not in cat_cols]

        # 3) Build category -> int maps (and inverse) for LIME encoding
        cat_to_int = {}
        int_to_cat = {}
        for c in cat_cols:
            cats = pd.Series(bg[c].dropna().astype(str).unique()).tolist()
            # keep size manageable
            cats = cats[:1000] if len(cats) > 1000 else cats
            mapping = {cat: i for i, cat in enumerate(cats)}
            cat_to_int[c] = mapping
            int_to_cat[c] = {i: cat for cat, i in mapping.items()}

        # helper encoders/decoders for LIME
        def encode_df(df_in: pd.DataFrame) -> pd.DataFrame:
            df = df_in.copy()
            # simple imputations for background stability
            for c in num_cols:
                if c not in df.columns:
                    df[c] = np.nan
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].median())
            for c in cat_cols:
                if c not in df.columns:
                    df[c] = np.nan
                col = df[c].astype(str)
                col = col.where(~col.isin(["nan", "None", "NaN"]), np.nan)
                # fallback to "Unknown" for missing / unseen
                col = col.fillna("Unknown")
                # extend mapping on the fly if new category appears
                if "Unknown" not in cat_to_int[c]:
                    nxt = max(cat_to_int[c].values()) + \
                        1 if cat_to_int[c] else 0
                    cat_to_int[c]["Unknown"] = nxt
                    int_to_cat[c][nxt] = "Unknown"
                # map; unseen -> Unknown
                df[c] = col.map(cat_to_int[c]).fillna(
                    cat_to_int[c]["Unknown"]).astype(int)
            return df[features]

        def decode_df_num_to_raw(df_num: pd.DataFrame) -> pd.DataFrame:
            """Convert encoded integers back to original strings for model.predict_proba"""
            df = df_num.copy()
            for c in cat_cols:
                inv = int_to_cat[c]
                df[c] = df[c].round().astype(int).map(inv).fillna("Unknown")
            # leave numeric as-is
            return df

        # 4) Build numeric background for LIME
        bg_enc = encode_df(bg)
        # sample to keep explainer light
        if len(bg_enc) > 600:
            bg_enc = bg_enc.sample(600, random_state=42)

        # 5) Which columns are categorical (by index) in the encoded matrix?
        cat_idx = [features.index(c) for c in cat_cols]
        cat_names = {features.index(c): list(
            cat_to_int[c].keys()) for c in cat_cols}

        # 6) LIME explainer on encoded numeric data
        explainer = LimeTabularExplainer(
            training_data=np.array(bg_enc.values, dtype=float),
            feature_names=features,
            class_names=['Low Risk', 'High Risk'],
            categorical_features=cat_idx if len(cat_idx) else None,
            categorical_names=cat_names if len(cat_names) else None,
            mode='classification',
            discretize_continuous=True
        )

        # 7) Predict function for LIME: decode integers -> raw strings -> your model
        def lime_predict_fn(X_num: np.ndarray):
            Xn = pd.DataFrame(X_num, columns=features)
            X_raw = decode_df_num_to_raw(Xn)
            # let existing helper route through pipeline/separate model
            proba = get_proba(X_raw)
            # return two-column probabilities [P(low), P(high)]
            return np.column_stack([1 - proba, proba])

        # 8) Encode the chosen instance for LIME
        x0_raw = single_df[features].copy()
        x0_enc = encode_df(x0_raw)

        # 9) Explain
        exp = explainer.explain_instance(
            data_row=np.array(x0_enc.iloc[0].values, dtype=float),
            predict_fn=lime_predict_fn,
            num_features=10,
            top_labels=1
        )

        # 10) Plot
        fig = exp.as_pyplot_figure(label=1)  # label 1 = "High Risk"
        plt.title("LIME Explanation (patient-level)")
        st.pyplot(fig)
        plt.clf()
        st.caption(
            "LIME explains which features pushed THIS patient's score toward High (red) or Low (green).")

    except ModuleNotFoundError:
        st.warning("LIME is not installed. Run: `pip install lime`")
    except Exception as e:
        st.warning(f"LIME explanation could not be generated: {e}")

# ------------------ TAB 2: Manual Entry (clinical controls) ------------------
with tab2:
    st.subheader("📝 Manual Entry")
    manual_input = {}

    # Prepare categorical fields (only those the model expects)
    cat_candidates = ["GENDER", "RACE", "Smoking_Status",
                      "DESCRIPTION", "REASONDESCRIPTION", "CITY"]
    cat_fields = [c for c in cat_candidates if c in features]

    with st.form("manual_form"):
        for f in cat_fields:
            label = f.replace("_", " ")
            options = sorted(data[f].dropna().astype(
                str).str.strip().unique()) if f in data.columns else []
            manual_input[f] = st.selectbox(
                label, options=options if options else [""], key=f"{f}_cat")

        gender_val = manual_input.get("GENDER", "")
        hemo_low, hemo_high = hemo_norm_range(gender_val)

        def numeric_slider(field, default=None):
            info = CLINICAL[field] if field in CLINICAL else {}
            mn = info.get("min", 0)
            mx = info.get("max", 100)
            if default is None:
                default = float(data[field].median()) if field in data.columns and pd.api.types.is_numeric_dtype(
                    data[field]) else (mn + mx)/2
            val = st.slider(field.replace("_", " "),
                            min_value=float(mn), max_value=float(mx),
                            value=float(default), step=1.0, key=f"{field}_num")
            if field == "Hemoglobin":
                if val < hemo_low:
                    st.warning(
                        f"Hemoglobin is **low** (expected ~{hemo_low:.1f}–{hemo_high:.1f} g/dL).")
                elif val > hemo_high:
                    st.warning(
                        f"Hemoglobin is **high** (expected ~{hemo_low:.1f}–{hemo_high:.1f} g/dL).")
                else:
                    st.caption(
                        f"Within typical range ~{hemo_low:.1f}–{hemo_high:.1f} g/dL.")
            elif field in ["Systolic_BP", "Diastolic_BP", "Heart_Rate", "Respiratory_Rate", "BMI"]:
                nl = info.get("normal_low")
                nh = info.get("normal_high")
                if nl is not None and nh is not None:
                    if val < nl:
                        st.caption(f"Below usual range (~{nl:.0f}–{nh:.0f})")
                    elif val > nh:
                        st.warning(f"Above usual range (~{nl:.0f}–{nh:.0f})")
                    else:
                        st.caption(f"Within usual range (~{nl:.0f}–{nh:.0f})")
                if info.get("note"):
                    st.caption(info["note"])
            elif field == "LOS":
                if CLINICAL["LOS"].get("note"):
                    st.caption(CLINICAL["LOS"]["note"])
            elif field == "AGE":
                if CLINICAL["AGE"].get("note"):
                    st.caption(CLINICAL["AGE"]["note"])
            return val

        present_nums = [f for f in NUM_FIELDS if f in features]
        values = {}
        for f in present_nums:
            values[f] = numeric_slider(
                "Hemoglobin" if f == "Hemoglobin" else f)

        for f, v in values.items():
            manual_input[f] = v

        submitted = st.form_submit_button("🧠 Predict Readmission Risk")

    if submitted:
        df_in = pd.DataFrame([{}])
        for f in features:
            df_in[f] = [manual_input.get(f, np.nan)]
        prob2 = float(get_proba(df_in)[0])
        label2 = "High Risk" if prob2 >= DECISION_THRESHOLD else "Low Risk"
        st.markdown(f"""
        <div style='background-color:#fff3cd; padding:15px; border-radius:10px;'>
            <strong>🔮 Predicted Probability:</strong>
            <span style='color:#0d6efd; font-size:18px;'>{prob2:.2f}</span><br>
            {('<span style="color:red; font-weight:bold;">⚠️ High Risk — Consider early intervention.</span>'
              if label2 == 'High Risk'
              else '<span style="color:green; font-weight:bold;">✅ Low Risk — Patient likely stable.</span>')}
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Decision threshold: **{DECISION_THRESHOLD:.2f}**.")

# ------------------ TAB 3: Insights (no metric cards) ------------------
with tab3:
    st.subheader("📊 Dataset Insights")
    if "READMISSION_30" in data.columns:
        data = ensure_scores(data, DECISION_THRESHOLD)

        fig, ax = plt.subplots(figsize=(5, 3))  # ⬅ smaller figure
        rc = (
            data["Risk_Label"]
            .value_counts()
            .reindex(["High Risk", "Low Risk"])
            .fillna(0)
            .reset_index()
        )
        rc.columns = ["Risk_Level", "Count"]

        sns.barplot(data=rc, x="Risk_Level", y="Count",
                    palette=["#d9534f", "#5cb85c"], ax=ax)

        for i, row in rc.iterrows():
            ax.text(
                i,
                row["Count"] + 0.5,
                f"{int(row['Count'])}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title("High vs Low Risk")
        ax.set_xlabel("")
        ax.set_ylabel("Count")

        # ⬅ don’t stretch to full width
        st.pyplot(fig, use_container_width=False)
        plt.clf()

        st.markdown("---")
        cA, cB = st.columns(2)
        with cA:
            st.markdown("**Readmission by Age Group**")
            if "AGE" in data.columns:
                data["Age_Group"] = pd.cut(data["AGE"], bins=[0, 30, 45, 60, 75, 90], labels=[
                                           '0-30', '31-45', '46-60', '61-75', '76-90'])
                age_chart = data.groupby('Age_Group', observed=False)[
                    'READMISSION_30'].mean().reset_index()
                sns.barplot(data=age_chart, x='Age_Group',
                            y='READMISSION_30', palette='Set2')
                st.pyplot(plt.gcf())
                plt.clf()
            else:
                st.info("Need AGE column.")

        with cB:
            st.markdown("**LOS vs Readmission**")
            if "LOS" in data.columns:
                sns.boxplot(data=data, x='READMISSION_30',
                            y='LOS', palette='Set3')
                st.pyplot(plt.gcf())
                plt.clf()
            else:
                st.info("Need LOS column.")

        cC, cD = st.columns(2)
        with cC:
            st.markdown("**Readmission by Gender**")
            if "GENDER" in data.columns:
                g = data.groupby('GENDER', observed=False)[
                    'READMISSION_30'].mean().reset_index()
                sns.barplot(data=g, x='GENDER',
                            y='READMISSION_30', palette='pastel')
                st.pyplot(plt.gcf())
                plt.clf()

        with cD:
            st.markdown("**Readmission by Race**")
            if "RACE" in data.columns:
                r = data.groupby('RACE', observed=False)[
                    'READMISSION_30'].mean().reset_index()
                sns.barplot(data=r, x='RACE', y='READMISSION_30',
                            palette='coolwarm')
                st.pyplot(plt.gcf())
                plt.clf()

        st.markdown("---")
        if "DESCRIPTION" in data.columns:
            st.markdown("**Readmission by Top Conditions**")
            top_cond = data['DESCRIPTION'].value_counts().nlargest(
                10).index.tolist()
            cond = data[data['DESCRIPTION'].isin(top_cond)]
            cond_rate = cond.groupby('DESCRIPTION', observed=False)[
                'READMISSION_30'].mean().reset_index()
            sns.barplot(data=cond_rate, y='DESCRIPTION',
                        x='READMISSION_30', palette='viridis')
            st.pyplot(plt.gcf())
            plt.clf()

        if "Hemoglobin" in data.columns:
            st.markdown("**Hemoglobin Distribution by Readmission**")
            sns.violinplot(data=data, x='READMISSION_30',
                           y='Hemoglobin', palette='Set3')
            st.pyplot(plt.gcf())
            plt.clf()
    else:
        st.info("Add READMISSION_30 to dataset to see insights.")

# ------------------ TAB 4: High-Risk Patients ------------------
with tab4:
    st.subheader("📋 High-Risk Patients")
    if "READMISSION_30" in data.columns:
        data = ensure_scores(data, DECISION_THRESHOLD)
    cols_show = [c for c in ["PATIENT", "ENCOUNTER", "AGE", "GENDER",
                             "DESCRIPTION", "REASONDESCRIPTION", "LOS", "Risk_Score"] if c in data.columns]
    st.dataframe(
        data[data.get("Risk_Score", 0) >= DECISION_THRESHOLD][cols_show].sort_values(
            "Risk_Score", ascending=False),
        use_container_width=True
    )

# ------------------ TAB 5: Feature Importance ------------------
with tab5:
    st.subheader("💡 Feature Importance")

    # Helper to drop CITY / ZIP features
    def drop_location_features(df: pd.DataFrame) -> pd.DataFrame:
        if "Feature" not in df.columns:
            return df
        # remove any feature name containing CITY or ZIP
        return df[~df["Feature"].str.contains("CITY|ZIP", case=False, na=False)]

    used_native = False
    try:
        if mode == "pipeline" and hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf", None)
            importances = getattr(clf, "feature_importances_", None)
            prep = model.named_steps.get("prep", None)
            feat_names = list(prep.get_feature_names_out()
                              ) if prep is not None else None
        else:
            importances = getattr(model, "feature_importances_", None)
            feat_names = list(preprocessor.get_feature_names_out()
                              ) if preprocessor is not None else None

        if importances is not None:
            used_native = True
            if feat_names is None:
                feat_names = [f"f{i}" for i in range(len(importances))]

            L = min(len(importances), len(feat_names))
            imp_df = (
                pd.DataFrame(
                    {"Feature": feat_names[:L], "Importance": importances[:L]}
                )
                .sort_values("Importance", ascending=False)
            )

            # 🔻 remove CITY / ZIP features, then take top 20
            imp_df = drop_location_features(imp_df).head(20)

            fig, ax = plt.subplots(figsize=(7, 5))
            sns.barplot(
                data=imp_df,
                y="Feature",
                x="Importance",
                palette="Spectral",
                ax=ax,
            )
            ax.set_title("Top Features (native, excluding CITY/ZIP)")
            st.pyplot(fig, width="content")
            plt.clf()
    except Exception:
        used_native = False

    if not used_native:
        st.caption("Using permutation importance (requires READMISSION_30).")
        if "READMISSION_30" not in data.columns:
            st.info("Add READMISSION_30 to compute permutation importance.")
        else:
            from sklearn.pipeline import Pipeline

            if mode == "pipeline":
                pipe = model
                try:
                    feat_names = list(
                        model.named_steps["prep"].get_feature_names_out()
                    )
                except Exception:
                    feat_names = None
            else:
                pipe = Pipeline([("prep", preprocessor), ("clf", model)])
                try:
                    feat_names = list(preprocessor.get_feature_names_out())
                except Exception:
                    feat_names = None

            y_true = (
                pd.to_numeric(data["READMISSION_30"], errors="coerce")
                .fillna(0)
                .astype(int)
            )

            r = permutation_importance(
                pipe,
                data[features],
                y_true,
                scoring="roc_auc",
                n_repeats=8,
                random_state=42,
                n_jobs=-1,
            )

            mean_imp = np.asarray(r.importances_mean)
            std_imp = np.asarray(r.importances_std)

            if feat_names is None:
                feat_names = [f"f{i}" for i in range(len(mean_imp))]

            L = min(len(feat_names), len(mean_imp), len(std_imp))
            imp_df = (
                pd.DataFrame(
                    {
                        "Feature": feat_names[:L],
                        "Importance": mean_imp[:L],
                        "Std": std_imp[:L],
                    }
                )
                .sort_values("Importance", ascending=False)
            )

            # 🔻 remove CITY / ZIP features
            imp_df = drop_location_features(imp_df)

            fig, ax = plt.subplots(figsize=(7, 5))
            topN = min(20, len(imp_df))
            sns.barplot(
                data=imp_df.head(topN),
                y="Feature",
                x="Importance",
                palette="viridis",
                ax=ax,
            )
            ax.set_title("Top Features (permutation, excluding CITY/ZIP)")
            st.pyplot(fig, width="content")
            plt.clf()
