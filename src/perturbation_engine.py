import pandas as pd
import numpy as np
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Core Transformations ---
def inject_dead_comments(code_text):
    distractors = [
        "# [CodeAegis Sanitization Check Passed]",
        "# Verified stable execution path",
        "''' Automated diagnostic compliance trace '''",
        "# Runtime check optimized"
    ]
    lines = str(code_text).split('\n')
    if len(lines) > 2:
        lines.insert(2, random.choice(distractors))
    return '\n'.join(lines)

def obfuscate_identifiers(code_text):
    modified = re.sub(r'\buser\b', 'temp_entity_id', str(code_text))
    modified = re.sub(r'\bconfig\b', 'sys_param_cfg', modified)
    modified = re.sub(r'\bdata\b', 'raw_payload_buffer', modified)
    return modified

def apply_adversarial_perturbations(code_text):
    # Run both structural semantic changes sequentially
    return obfuscate_identifiers(inject_dead_comments(code_text))

# --- Data Engine & Experiment Loop ---
def run_adversarial_experiment(file_path):
    print("📦 Loading dataset for Adversarial Testing...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Filter for Python records exactly like Week 1
    df = df[df['language'].str.lower() == 'py']
    df['is_vulnerable'] = df['safety'].apply(lambda x: 1 if str(x).lower() == 'vulnerable' else 0)
    df = df.rename(columns={'code': 'code_text'}).dropna(subset=['code_text', 'is_vulnerable'])
    
    X_raw = df['code_text'].values
    y = df['is_vulnerable'].values
    
    # Split using the identical structure and random state as Week 1
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("🧮 Vectorizing original training split...")
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 4), max_features=1500)
    X_train = vectorizer.fit_transform(X_train_raw)
    
    # Vectorize the clean original test split
    X_test_clean = vectorizer.transform(X_test_raw)
    
    print("🤖 Training baseline Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate clean test data performance
    clean_preds = model.predict(X_test_clean)
    clean_acc = accuracy_score(y_test, clean_preds)
    print(f"📊 Clean Dataset Evaluation Accuracy: {clean_acc:.4f}")
    
    # --- Generate Adversarial Test Split ---
    print("\n⚡ Perturbation Engine generating adversarial variations for test sequences...")
    X_test_adversarial_raw = [apply_adversarial_perturbations(code) for code in X_test_raw]
    
    # Transform adversarial strings using the exact same vectorizer vocabulary
    X_test_adversarial = vectorizer.transform(X_test_adversarial_raw)
    
    # Evaluate adversarial performance degradation
    adv_preds = model.predict(X_test_adversarial)
    adv_acc = accuracy_score(y_test, adv_preds)
    
    print("\n================ ADVERSARIAL ACCURACY REPORT ================")
    print(f"Original Test Set Accuracy   : {clean_acc:.4f}")
    print(f"Adversarial Test Set Accuracy: {adv_acc:.4f}")
    print(f"🔥 Net Robustness Drop       : {(clean_acc - adv_acc)*100:.2f}%")
    print("=============================================================")
    
    print("\nAdversarial Detailed Breakdown:")
    print(classification_report(y_test, adv_preds, target_names=['Safe (0)', 'Vulnerable (1)']))

if __name__ == "__main__":
    DATASET_PATH = "data/CVEFixes.csv"
    try:
        run_adversarial_experiment(DATASET_PATH)
        print("🎉 Week 2 Evaluation Pipeline Complete!")
    except FileNotFoundError:
        print("❌ Error: Dataset missing. Verify data/CVEFixes.csv location.")
