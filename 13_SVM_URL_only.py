# File: 13_SVM_URL_only.py
import pandas as pd
import re
import numpy as np
import gc
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score, average_precision_score, log_loss
)
from datasets import load_dataset
import warnings
from urllib.parse import urlparse

warnings.filterwarnings("ignore")
np.random.seed(42)

print("="*80)
print("SVM (URL-only) - 10-Fold CV with Early Stopping")
print("="*80)

# ====================== LOAD DATASET ======================
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset)

label_col = 'labels' if 'labels' in df.columns else 'label'
df = df[df[label_col].isin([0, 1])].reset_index(drop=True)
df = df.rename(columns={label_col: "label", "content": "email_text"})

def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else None

df['url'] = df['email_text'].apply(extract_first_url)

phishing_urls = df[df['label'] == 1]['url'].dropna().tolist()
legit_urls = df[df['label'] == 0]['url'].dropna().tolist()

import random
random.seed(42)

def get_synthetic_url(label):
    if label == 1 and phishing_urls: return random.choice(phishing_urls)
    elif label == 0 and legit_urls: return random.choice(legit_urls)
    return ""

df['url'] = df.apply(lambda row: row['url'] if pd.notna(row['url']) else get_synthetic_url(row['label']), axis=1)

# ====================== LEXICAL FEATURES ======================
def extract_hannousse_style_url_features(url):
    if not isinstance(url, str) or not url.strip():
        return {f: 0.0 for f in ['length_url','length_hostname','ip','nb_dots','nb_hyphens','nb_at','nb_qm',
                                 'nb_and','nb_or','nb_eq','nb_underscore','nb_tilde','nb_percent','nb_slash',
                                 'nb_star','nb_colon','nb_comma','nb_semicolumn','nb_dollar','nb_space',
                                 'nb_www','nb_com','nb_dslash','http_in_path','https_token','ratio_digits_url',
                                 'ratio_digits_host','punycode','shortening_service','prefix_suffix',
                                 'abnormal_subdomain','having_ip_address','having_at_symbol',
                                 'double_slash_redirecting','phish_hints']}
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    path = parsed.path or ''
    f = {}
    f['length_url'] = len(url)
    f['length_hostname'] = len(hostname)
    f['ip'] = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname) else 0
    f['nb_dots'] = url.count('.')
    f['nb_hyphens'] = url.count('-')
    f['nb_at'] = url.count('@')
    f['nb_qm'] = url.count('?')
    f['nb_and'] = url.count('&')
    f['nb_or'] = url.count('|')
    f['nb_eq'] = url.count('=')
    f['nb_underscore'] = url.count('_')
    f['nb_tilde'] = url.count('~')
    f['nb_percent'] = url.count('%')
    f['nb_slash'] = url.count('/')
    f['nb_star'] = url.count('*')
    f['nb_colon'] = url.count(':')
    f['nb_comma'] = url.count(',')
    f['nb_semicolumn'] = url.count(';')
    f['nb_dollar'] = url.count('$')
    f['nb_space'] = url.count(' ')
    f['nb_www'] = 1 if 'www.' in url.lower() else 0
    f['nb_com'] = url.count('.com')
    f['nb_dslash'] = 1 if '//' in url[8:] else 0
    f['http_in_path'] = 1 if 'http' in path.lower() else 0
    f['https_token'] = 1 if url.startswith('https') else 0
    f['ratio_digits_url'] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
    f['ratio_digits_host'] = sum(c.isdigit() for c in hostname) / len(hostname) if len(hostname) > 0 else 0
    f['punycode'] = 1 if 'xn--' in hostname else 0
    f['shortening_service'] = 1 if any(s in url.lower() for s in ['bit.ly','tinyurl','t.co','goo.gl','ow.ly']) else 0
    f['prefix_suffix'] = 1 if '-' in hostname.split('.')[0] else 0
    f['abnormal_subdomain'] = 1 if hostname.count('.') > 3 else 0
    f['having_ip_address'] = f['ip']
    f['having_at_symbol'] = f['nb_at']
    f['double_slash_redirecting'] = f['nb_dslash']
    phish_keywords = ['login','secure','account','bank','paypal','update','verify','password','signin','confirm','alert']
    f['phish_hints'] = sum(1 for kw in phish_keywords if kw in url.lower())
    return f

print("Extracting lexical features...")
url_features_df = pd.DataFrame([extract_hannousse_style_url_features(u) for u in df['url']])
X = url_features_df.values.astype(np.float32)
y = df['label'].values

# ====================== METRICS ======================
def compute_all_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='binary', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='binary', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='binary', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["avg_precision"] = average_precision_score(y_true, y_prob)
        metrics["log_loss"] = log_loss(y_true, y_prob)
    return metrics

# ====================== 10-FOLD WITH EARLY STOPPING ======================
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"\n--- Fold {fold+1}/10 ---")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    max_epochs = 30

    for epoch in range(max_epochs):
        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_train)[:, 1]
        train_loss = log_loss(y_train, y_prob)

        print(f"  Epoch {epoch+1:2d}/{max_epochs} - Train Log Loss: {train_loss:.4f}")

        if train_loss < best_loss - 1e-5:   # small tolerance
            best_loss = train_loss
            patience_counter = 0
            print(f"    → New best loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"    → No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"    Early stopping triggered at epoch {epoch+1}")
            break

    # Final evaluation
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    metrics = compute_all_metrics(y_val, y_pred, y_prob)
    fold_results.append(metrics)

    print(f"  Fold {fold+1} - Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics.get('roc_auc',0):.4f}")

    gc.collect()

# ====================== FINAL RESULTS ======================
avg_metrics = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
print(f"\nFinal Average for SVM (URL-only):")
for k, v in avg_metrics.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

result_df = pd.DataFrame([avg_metrics])
result_df.to_csv("13_SVM_URL_only_results.csv", index=False)
print("\nResults saved to 13_SVM_URL_only_results.csv")
