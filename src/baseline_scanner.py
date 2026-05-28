import pandas as pd
import numpy as np
print("🔄 Step 1: Loading Scikit-Learn components...")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
print("✅ Step 2: All ML packages loaded successfully!")

def load_and_preprocess_data(file_path):
    print("\n📦 Loading dataset from CodeAegis data vault...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"📊 Total raw dataset records found: {len(df)}")
    
    # Filter for Python using the correct 'py' string found in your analysis
    if 'language' in df.columns:
        df = df[df['language'].str.lower() == 'py']
        print(f"🐍 Filtered for Python ('py') files. Remaining rows: {len(df)}")
        
    if 'safety' in df.columns:
        df['is_vulnerable'] = df['safety'].apply(lambda x: 1 if str(x).lower() == 'vulnerable' else 0)
    else:
        raise KeyError("❌ Error: Could not find 'safety' label column.")

    if 'code' in df.columns:
        df = df.rename(columns={'code': 'code_text'})
    else:
        raise KeyError("❌ Error: Could not find 'code' text column.")

    df = df.dropna(subset=['code_text', 'is_vulnerable'])
    print(f"✅ Preprocessing done. Vulnerable samples: {sum(df['is_vulnerable'])} | Safe samples: {len(df) - sum(df['is_vulnerable'])}")
    return df

def train_baseline_scanner(df):
    print("\n⚡ Initializing Character Level TF-IDF Vectorizer...")
    # Using 3-4 character n-grams to catch syntax structures like loops, spaces, and brackets
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 4), max_features=1500)
    
    print("🧮 Converting Python source code into feature matrices...")
    X = vectorizer.fit_transform(df['code_text'])
    y = df['is_vulnerable'].values
    
    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"🤖 Training baseline Random Forest Classifier (Samples: {X_train.shape[0]})...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Run evaluation
    predictions = model.predict(X_test)
    print("\n================ BASELINE PERFORMANCE REPORT ================")
    print(classification_report(y_test, predictions, target_names=['Safe (0)', 'Vulnerable (1)']))
    print(f"Overall Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("=============================================================")
    return model

if __name__ == "__main__":
    DATASET_PATH = "data/CVEFixes.csv" 
    try:
        data = load_and_preprocess_data(DATASET_PATH)
        if len(data) == 0:
            print("❌ Error: No samples remaining after filtering.")
        else:
            train_baseline_scanner(data)
            print("🎉 Week 1 Milestone Complete!")
    except FileNotFoundError:
        print(f"❌ Error: Could not find dataset at {DATASET_PATH}.")
