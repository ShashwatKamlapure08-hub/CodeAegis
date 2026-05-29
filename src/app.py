import streamlit as st
import pandas as pd
import numpy as np
import re
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# --- Page Adjustments ---
st.set_page_config(page_title="CodeAegis Shield Dashboard", page_icon="🛡️", layout="wide")

# --- Persistent ML Framework Initialization (Cached for Speed) ---
@st.cache_resource
def initialize_and_train_engines():
    # 1. Fallback / Dataset Routing Check
    # Points directly to the lightweight pre-filtered slice for cloud deployment
    optimized_path = "src/CVEFixes_python_only.csv"
    legacy_path = "data/CVEFixes.csv"
    
    if os.path.exists(optimized_path):
        df = pd.read_csv(optimized_path)
    elif os.path.exists(legacy_path):
        # Fallback for local testing if the optimized file isn't built yet
        df = pd.read_csv(legacy_path, low_memory=False)
        df = df[df['language'].str.lower() == 'py']
        df['is_vulnerable'] = df['safety'].apply(lambda x: 1 if str(x).lower() == 'vulnerable' else 0)
        df = df.rename(columns={'code': 'code_text'}).dropna(subset=['code_text', 'is_vulnerable'])
    else:
        raise FileNotFoundError(
            "Could not locate a valid training dataset slice. "
            "Ensure 'src/CVEFixes_python_only.csv' is generated and pushed to your repo."
        )
    
    X_raw = df['code_text'].astype(str).values
    y = df['is_vulnerable'].values
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Train Baseline Setup
    vec_baseline = TfidfVectorizer(analyzer='char', ngram_range=(3, 4), max_features=1500)
    X_train_base = vec_baseline.fit_transform(X_train_raw)
    model_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model_baseline.fit(X_train_base, y_train)
    
    # 3. Train Patched Setup (With Data Augmentation)
    def quick_perturb(text):
        distractors = ["# [CodeAegis Sanitization Check Passed]", "# Verified stable execution path"]
        lines = str(text).split('\n')
        if len(lines) > 2:
            lines.insert(2, random.choice(distractors))
        mod = '\n'.join(lines)
        mod = re.sub(r'\buser\b', 'temp_entity_id', mod)
        mod = re.sub(r'\bconfig\b', 'sys_param_cfg', mod)
        mod = re.sub(r'\bdata\b', 'raw_payload_buffer', mod)
        return mod

    X_train_aug_raw = [quick_perturb(code) for code in X_train_raw]
    X_train_combined = np.concatenate([X_train_raw, X_train_aug_raw])
    y_train_combined = np.concatenate([y_train, y_train])
    
    vec_patched = TfidfVectorizer(analyzer='char', ngram_range=(3, 4), max_features=1500)
    X_train_patched = vec_patched.fit_transform(X_train_combined)
    model_patched = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model_patched.fit(X_train_patched, y_train_combined)
    
    return vec_baseline, model_baseline, vec_patched, model_patched

# Load ML systems background tier
with st.spinner("🔮 Training and calibrating CodeAegis Core Defense Engines... Please wait."):
    vec_base, model_base, vec_patch, model_patch = initialize_and_train_engines()

# --- Adversarial Transformation Rules ---
def apply_live_camouflage(code_text):
    lines = str(code_text).split('\n')
    if len(lines) > 2:
         lines.insert(2, "# [Adversarial Camouflage Injected By Attacker]")
    modified = '\n'.join(lines)
    modified = re.sub(r'\buser\b', 'camouflaged_entity_var', modified)
    modified = re.sub(r'\bconfig\b', 'masked_config_params', modified)
    modified = re.sub(r'\bdata\b', 'obfuscated_payload', modified)
    return modified

# --- Header Architecture ---
st.title("🛡️ CodeAegis: Adversarial Robustness Analyzer")
st.markdown("---")

# --- Main Layout Splitting ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input Source Sequence")
    default_snippet = """def standard_authenticator(user, config):\n    data = user.get_payload()\n    # Processing credential metrics\n    return data"""
    user_code = st.text_area("Paste Python code context here:", value=default_snippet, height=180)

with col2:
    st.subheader("🎭 Automated Adversarial Camouflage")
    if st.checkbox("Apply Structural Perturbations (Rename Variables & Inject Dead Comments)", value=True):
        processed_code = apply_live_camouflage(user_code)
    else:
        processed_code = user_code
    st.code(processed_code, language="python")

st.markdown("---")
st.subheader("🔍 Comparative Security Scanning Diagnostics")

scan_col1, scan_col2 = st.columns(2)

with scan_col1:
    st.info("⚠️ Baseline Engine (Week 1)")
    features_base = vec_base.transform([processed_code])
    pred_base = model_base.predict(features_base)[0]
    prob_base = model_base.predict_proba(features_base)[0]
    
    st.metric(label="Predicted Safety Metric", value="🔴 Vulnerable" if pred_base == 1 else "🟢 Safe")
    st.write(f"Confidence Level: **{prob_base[pred_base]*100:.2f}%**")

with scan_col2:
    st.success("🛡️ Adversarially Patched Engine (Week 3)")
    features_patch = vec_patch.transform([processed_code])
    pred_patch = model_patch.predict(features_patch)[0]
    prob_patch = model_patch.predict_proba(features_patch)[0]
    
    st.metric(label="Predicted Safety Metric", value="🔴 Vulnerable" if pred_patch == 1 else "🟢 Safe")
    st.write(f"Confidence Level: **{prob_patch[pred_patch]*100:.2f}%**")