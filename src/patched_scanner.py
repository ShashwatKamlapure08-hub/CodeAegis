import pandas as pd
import numpy as np
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Core Transformations for Data Augmentation ---
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
    return obfuscate_identifiers(inject_dead_comments(code_text))

# --- Defensive Robustness Pipeline ---
def run_defensive_training(file_path):
    print("📦 Step 1: Loading raw dataset from CodeAegis data vault...")
    df = pd.read_csv(file_path, low_memory=False)
    df = df[df['language'].str.lower() == 'py']
    df['is_vulnerable'] = df['safety'].apply(lambda x: 1 if str(x).lower() == 'vulnerable' else 0)
    df = df.rename(columns={'code': 'code_text'}).dropna(subset=['code_text', 'is_vulnerable'])
    
    X_raw = df['code_text'].values
    y = df['is_vulnerable'].values
    
    # Split identical to Week 1 & Week 2
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("⚡ Step 2: Generating adversarial training mutations for augmentation...")
    # Duplicate the training data and apply camouflage modifications
    X_train_augmented_raw = [apply_adversarial_perturbations(code) for code in X_train_raw]
    
    # Combine original clean training text with adversarial training text
    X_train_combined_raw = np.concatenate([X_train_raw, X_train_augmented_raw])
    y_train_combined = np.concatenate([y_train, y_train])
    
    print(f"📊 Combined Training Matrix expanded from {len(X_train_raw)} to {len(X_train_combined_raw)} rows.")
    
    print("🧮 Step 3: Fitting Vectorizer on the expanded vocabulary...")
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 4), max_features=1500)
    X_train_final = vectorizer.fit_transform(X_train_combined_raw)
    
    # Process test datasets
    X_test_clean = vectorizer.transform(X_test_raw)
    X_test_adversarial_raw = [apply_adversarial_perturbations(code) for code in X_test_raw]
    X_test_adversarial = vectorizer.transform(X_test_adversarial_raw)
    
    print("🤖 Step 4: Training the Adversarially Patched Random Forest Classifier...")
    patched_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    patched_model.fit(X_train_final, y_train_combined)
    
    # Evaluate Patched Model Performance
    patched_clean_preds = patched_model.predict(X_test_clean)
    patched_adv_preds = patched_model.predict(X_test_adversarial)
    
    patched_clean_acc = accuracy_score(y_test, patched_clean_preds)
    patched_adv_acc = accuracy_score(y_test, patched_adv_preds)
    
    print("\n================ DEFENSIVE PATCH PERFORMANCE REPORT ================")
    print(f"Patched Model on Clean Test Set Accuracy      : {patched_clean_acc:.4f}")
    print(f"Patched Model on Adversarial Test Set Accuracy: {patched_adv_acc:.4f}")
    print(f"🛡️ Post-Patch Robustness Delta                : {(patched_clean_acc - patched_adv_acc)*100:.2f}%")
    print("=====================================================================")
    
    print("\nPatched Model Adversarial Detailed Breakdown:")
    print(classification_report(y_test, patched_adv_preds, target_names=['Safe (0)', 'Vulnerable (1)']))

if __name__ == "__main__":
    DATASET_PATH = "data/CVEFixes.csv"
    try:
        run_defensive_training(DATASET_PATH)
        print("🎉 Week 3 Defensive Retraining Milestone Complete!")
    except FileNotFoundError:
        print("❌ Error: Core data/CVEFixes.csv file missing.")
