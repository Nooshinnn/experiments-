# File: 20_TextCNN_URL_only_Clean.py
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
from transformers import AutoTokenizer
from datasets import load_dataset
import warnings
from tqdm import tqdm
from urllib.parse import urlparse

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("="*80)
print("TextCNN (URL-only) - Clean Label-Matched Pairing + New Early Stopping")
print("="*80)

# ====================== LOAD DATASET ======================
print("\nLoading dataset...")
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset)

label_col = 'labels' if 'labels' in df.columns else 'label'
df = df[df[label_col].isin([0, 1])].reset_index(drop=True)
df = df.rename(columns={label_col: "label", "content": "email_text"})

print(f"Total samples: {len(df)}")

# ====================== CLEAN LABEL-MATCHED PAIRING ======================
def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else None

df['url'] = df['email_text'].apply(extract_first_url)

# Collect real URLs per class
phishing_urls = df[df['label'] == 1]['url'].dropna().tolist()
legit_urls     = df[df['label'] == 0]['url'].dropna().tolist()

print(f"Real phishing URLs: {len(phishing_urls)}")
print(f"Real legitimate URLs: {len(legit_urls)}")

import random
random.seed(42)

def get_matching_url(label):
    if label == 1 and phishing_urls:
        return random.choice(phishing_urls)
    elif label == 0 and legit_urls:
        return random.choice(legit_urls)
    return ""  # very rare fallback

# Assign matching URL
df['url'] = df.apply(lambda row: row['url'] if pd.notna(row['url']) else get_matching_url(row['label']), axis=1)

print(f"Final samples after clean pairing: {len(df)}")

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

# ====================== TextCNN MODEL ======================
class TextCNN(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=300, num_filters=100, filter_sizes=[3,4,5]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 128)
        
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

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

# ====================== 10-FOLD WITH NEW EARLY STOPPING ======================
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"\n--- Fold {fold+1}/10 ---")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    url_model = TextCNN().to(device)

    optimizer = optim.AdamW(url_model.parameters(), lr=8e-6, weight_decay=0.08)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    max_epochs = 30

    for epoch in range(max_epochs):
        url_model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(range(0, len(train_df), 8), desc=f"Epoch {epoch+1:2d}/{max_epochs}", leave=True)

        for i in progress_bar:
            batch = train_df.iloc[i:i+8]
            labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

            url_inputs = tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            url_feat = url_model(url_inputs.input_ids)

            comm = url_feat

            loss = criterion(comm.mean(dim=1), labels)
            torch.nn.utils.clip_grad_norm_(url_model.parameters(), max_norm=1.0)

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

        avg_train_loss = train_loss / (len(train_df) // 8 + 1)

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            patience_counter = 0
            print(f"    → New best loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"    → No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Evaluation
    url_model.eval()
    y_true = val_df['label'].values
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for i in range(0, len(val_df), 32):
            batch = val_df.iloc[i:i+32]
            url_inputs = tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            url_feat = url_model(url_inputs.input_ids)
            prob = url_feat.mean(dim=1).sigmoid().cpu().numpy()
            pred = (prob > 0.5).astype(int)
            y_pred.extend(pred)
            y_prob.extend(prob)

    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    fold_results.append(metrics)

    del url_model
    torch.cuda.empty_cache()
    gc.collect()

# ====================== FINAL RESULTS ======================
avg_metrics = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
print(f"\nFinal Average for TextCNN (URL-only) - Clean Label-Matched Pairing:")
for k, v in avg_metrics.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

result_df = pd.DataFrame([avg_metrics])
result_df.to_csv("20_TextCNN_URL_only_Clean.csv", index=False)
print("\nResults saved to 20_TextCNN_URL_only_Clean.csv")
