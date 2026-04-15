# File: 02_BERT_Email_RandomForest_URL.py
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score, average_precision_score, log_loss
)
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import warnings
from tqdm import tqdm
from urllib.parse import urlparse

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================== GPU CHECK ======================
print("="*80)
print("GPU STATUS CHECK")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("="*80)

# ====================== LOAD LARGE DATASET ======================
print("\nLoading large training dataset...")
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset)

label_col = 'labels' if 'labels' in df.columns else 'label'
df = df[df[label_col].isin([0, 1])].reset_index(drop=True)
df = df.rename(columns={label_col: "label", "content": "email_text"})

print(f"Training email samples: {len(df)}")

# ====================== CREATE EMAIL-URL PAIRS ======================
def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else None

df['url'] = df['email_text'].apply(extract_first_url)

phishing_urls = df[df['label'] == 1]['url'].dropna().tolist()
legit_urls = df[df['label'] == 0]['url'].dropna().tolist()

import random
random.seed(42)

def get_synthetic_url(label):
    if label == 1 and phishing_urls:
        return random.choice(phishing_urls)
    elif label == 0 and legit_urls:
        return random.choice(legit_urls)
    return ""

df['url'] = df.apply(lambda row: row['url'] if pd.notna(row['url']) else get_synthetic_url(row['label']), axis=1)

print(f"Final paired samples: {len(df)}")

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

# ====================== MODELS ======================
class BertWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.model.config.hidden_size, 128)
        self.dropout = nn.Dropout(0.3)
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.fc(pooled)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ====================== ALL METRICS ======================
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

# ====================== 10-FOLD TRAINING ======================
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"\n--- Fold {fold+1}/10 ---")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_url_feat = url_features_df.iloc[train_idx].reset_index(drop=True)
    val_url_feat = url_features_df.iloc[val_idx].reset_index(drop=True)

    email_model = BertWrapper().to(device)

    # ====================== TRAINING ======================
    optimizer = optim.AdamW(email_model.parameters(), lr=8e-6, weight_decay=0.08)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(20):
        email_model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(range(0, len(train_df), 8), desc=f"Epoch {epoch+1:2d}/20", leave=True)

        for i in progress_bar:
            batch = train_df.iloc[i:i+8]
            labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

            email_inputs = tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            email_feat = email_model(email_inputs.input_ids, email_inputs.attention_mask)

            # For RandomForest we only use lexical features, so no URL model needed during training
            comm = email_feat   # placeholder

            loss = criterion(comm.mean(dim=1), labels)
            torch.nn.utils.clip_grad_norm_(email_model.parameters(), max_norm=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (comm.mean(dim=1).sigmoid() > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

            progress_bar.set_postfix({
                'Train Loss': f'{train_loss / (i//8 + 1):.4f}',
                'Train Acc ': f'{train_correct / train_total:.4f}'
            })

        if epoch == 2 and hasattr(email_model, 'model'):
            for param in email_model.model.parameters():
                param.requires_grad = True
            print("    → Unfreezing base layers")

        if epoch >= 8 and train_loss > 0.9:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # ====================== EVALUATION ======================
    email_model.eval()

    y_true = val_df['label'].values
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for i in range(0, len(val_df), 32):
            batch = val_df.iloc[i:i+32]
            bs = len(batch)

            email_inputs = tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            email_feat = email_model(email_inputs.input_ids, email_inputs.attention_mask)
            email_np = email_feat.detach().cpu().numpy()

            # RandomForest uses ONLY lexical features
            url_feat_np = val_url_feat.iloc[i:i+32].values.astype(np.float32)

            model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            model.fit(train_url_feat.values, train_df['label'].values)
            pred = model.predict(url_feat_np)
            prob = model.predict_proba(url_feat_np)[:, 1]

            y_pred.extend(pred)
            y_prob.extend(prob)

    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    fold_results.append(metrics)

    del email_model
    torch.cuda.empty_cache()
    gc.collect()

# ====================== FINAL RESULTS ======================
avg_metrics = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
print(f"\nFinal Average for BERT (Email) + RandomForest (URL):")
print(f"  Accuracy: {avg_metrics['accuracy']:.4f}")
print(f"  Balanced Accuracy: {avg_metrics['balanced_accuracy']:.4f}")
print(f"  Precision: {avg_metrics['precision']:.4f}")
print(f"  Recall: {avg_metrics['recall']:.4f}")
print(f"  F1: {avg_metrics['f1']:.4f}")
print(f"  F1 Macro: {avg_metrics['f1_macro']:.4f}")
print(f"  F1 Weighted: {avg_metrics['f1_weighted']:.4f}")
print(f"  ROC-AUC: {avg_metrics.get('roc_auc',0):.4f}")
print(f"  Avg Precision: {avg_metrics.get('avg_precision',0):.4f}")
print(f"  MCC: {avg_metrics['mcc']:.4f}")
print(f"  Cohen's Kappa: {avg_metrics['cohen_kappa']:.4f}")
print(f"  Log Loss: {avg_metrics.get('log_loss', np.nan):.4f}")

result_df = pd.DataFrame([avg_metrics])
result_df.to_csv("02_BERT_Email_RandomForest_URL_results.csv", index=False)
print("\nResults saved to 02_BERT_Email_RandomForest_URL_results.csv")
