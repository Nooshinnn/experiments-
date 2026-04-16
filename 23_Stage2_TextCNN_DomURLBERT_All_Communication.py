# File: 23_Stage2_TextCNN_DomURLBERT_All_Communication.py
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

print("="*90)
print("Stage 2: TextCNN (Email) + DomURLBERT (URL) - All Communication Styles")
print("="*90)

# ====================== LOAD DATASET ======================
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset)

label_col = 'labels' if 'labels' in df.columns else 'label'
df = df[df[label_col].isin([0, 1])].reset_index(drop=True)
df = df.rename(columns={label_col: "label", "content": "email_text"})

# ====================== CLEAN LABEL-MATCHED PAIRING ======================
def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else None

df['url'] = df['email_text'].apply(extract_first_url)

phishing_urls = df[df['label'] == 1]['url'].dropna().tolist()
legit_urls     = df[df['label'] == 0]['url'].dropna().tolist()

import random
random.seed(42)

def get_matching_url(label):
    if label == 1 and phishing_urls:
        return random.choice(phishing_urls)
    elif label == 0 and legit_urls:
        return random.choice(legit_urls)
    return ""

df['url'] = df.apply(lambda row: row['url'] if pd.notna(row['url']) else get_matching_url(row['label']), axis=1)

print(f"Final samples with clean pairing: {len(df)}")

# ====================== LEXICAL FEATURES (for reference) ======================
def extract_hannousse_style_url_features(url):
    # (same as before - kept for completeness)
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

url_features_df = pd.DataFrame([extract_hannousse_style_url_features(u) for u in df['url']])

# ====================== MODELS ======================
class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30522, 300)
        self.convs = nn.ModuleList([nn.Conv1d(300, 100, k) for k in [3,4,5]])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(300, 128)
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)

class DomURLBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("amahdaouy/DomURLs_BERT")
        self.fc = nn.Linear(768, 128)
        self.dropout = nn.Dropout(0.3)
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.fc(pooled)

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

# ====================== COMMUNICATION MODULES ======================
class SimpleConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    def forward(self, email_feat, url_feat):
        comm = torch.cat([email_feat, url_feat], dim=1)
        return self.fc(comm).squeeze(-1)

class WeightedScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.5, 0.5]))
    def forward(self, email_score, url_score):
        w = torch.softmax(self.weight, dim=0)
        return w[0] * email_score + w[1] * url_score

class GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
    def forward(self, email_feat, url_feat):
        comm = torch.cat([email_feat, url_feat], dim=1)
        gate = self.gate(comm)
        email_score = torch.sigmoid(torch.mean(email_feat, dim=1))
        url_score = torch.sigmoid(torch.mean(url_feat, dim=1))
        return gate[:, 0] * email_score + gate[:, 1] * url_score

class CrossAttention(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.fc = nn.Linear(dim*2, 1)
    def forward(self, email_feat, url_feat):
        Q = self.query(email_feat)
        K = self.key(url_feat)
        V = self.value(url_feat)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2,-1)) / np.sqrt(128), dim=-1)
        attended = torch.matmul(attn, V)
        comm = torch.cat([email_feat, attended], dim=1)
        return self.fc(comm).squeeze(-1)

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

# ====================== TRAINING FUNCTION ======================
def train_and_evaluate(communication_name, comm_module):
    print(f"\n=== Training with {communication_name} ===")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"  Fold {fold+1}/10")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        email_model = TextCNN().to(device)
        url_model = DomURLBERT().to(device)
        comm_model = comm_module().to(device)

        optimizer = optim.AdamW(list(email_model.parameters()) + 
                                list(url_model.parameters()) + 
                                list(comm_model.parameters()), lr=8e-6, weight_decay=0.08)
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        max_epochs = 30

        for epoch in range(max_epochs):
            email_model.train()
            url_model.train()
            comm_model.train()

            train_loss = 0.0
            train_correct = 0
            train_total = 0

            progress_bar = tqdm(range(0, len(train_df), 8), desc=f"Epoch {epoch+1:2d}/{max_epochs}", leave=False)

            for i in progress_bar:
                batch = train_df.iloc[i:i+8]
                labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

                email_inputs = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                url_inputs = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

                email_feat = email_model(email_inputs.input_ids)
                url_feat = url_model(url_inputs.input_ids, url_inputs.attention_mask)

                # Different communication styles
                if communication_name == "Simple Concat":
                    logits = comm_model(email_feat, url_feat)
                elif communication_name == "Weighted Score":
                    email_score = torch.sigmoid(torch.mean(email_feat, dim=1))
                    url_score = torch.sigmoid(torch.mean(url_feat, dim=1))
                    logits = comm_model(email_score, url_score)
                elif communication_name in ["Gated Fusion", "Cross Attention"]:
                    logits = comm_model(email_feat, url_feat)

                loss = criterion(logits, labels)
                torch.nn.utils.clip_grad_norm_(list(email_model.parameters()) + 
                                               list(url_model.parameters()) + 
                                               list(comm_model.parameters()), max_norm=1.0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = (logits.sigmoid() > 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total += len(labels)

                progress_bar.set_postfix({'Train Loss': f'{train_loss / (i//8 + 1):.4f}'})

            avg_train_loss = train_loss / (len(train_df) // 8 + 1)

            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        # Evaluation
        email_model.eval()
        url_model.eval()
        comm_model.eval()

        y_true = val_df['label'].values
        y_pred = []
        y_prob = []

        with torch.no_grad():
            for i in range(0, len(val_df), 32):
                batch = val_df.iloc[i:i+32]
                email_inputs = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                url_inputs = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

                email_feat = email_model(email_inputs.input_ids)
                url_feat = url_model(url_inputs.input_ids, url_inputs.attention_mask)

                if communication_name == "Simple Concat":
                    logits = comm_model(email_feat, url_feat)
                elif communication_name == "Weighted Score":
                    email_score = torch.sigmoid(torch.mean(email_feat, dim=1))
                    url_score = torch.sigmoid(torch.mean(url_feat, dim=1))
                    logits = comm_model(email_score, url_score)
                else:
                    logits = comm_model(email_feat, url_feat)

                prob = logits.sigmoid().cpu().numpy()
                pred = (prob > 0.5).astype(int)

                y_pred.extend(pred)
                y_prob.extend(prob)

        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        fold_results.append(metrics)

        del email_model, url_model, comm_model
        torch.cuda.empty_cache()
        gc.collect()

    avg_metrics = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
    print(f"\n=== {communication_name} Results ===")
    for k, v in avg_metrics.items():
        print(f"  {k.replace('_', ' ').title()}: {v:.4f}")
    return avg_metrics

# ====================== RUN ALL COMMUNICATION STYLES ======================
communication_styles = {
    "No Communication (Independent Avg)": None,   # placeholder
    "Simple Concat": SimpleConcat,
    "Weighted Score": WeightedScore,
    "Gated Fusion": GatedFusion,
    "Cross Attention": CrossAttention
}

results = {}
for name, module in communication_styles.items():
    if name == "No Communication (Independent Avg)":
        # Simple baseline: average of independent predictions (you can implement later if needed)
        print(f"\nSkipping {name} for now (can be added)")
        continue
    results[name] = train_and_evaluate(name, module)

print("\n=== All Communication Styles Comparison Complete ===")
print("You can now compare the results above to choose the best one for final Stage 2.")

# Optional: Save all results to CSV
comparison_df = pd.DataFrame(results).T
comparison_df.to_csv("23_Stage2_Communication_Comparison.csv")
print("Comparison table saved to 23_Stage2_Communication_Comparison.csv")
