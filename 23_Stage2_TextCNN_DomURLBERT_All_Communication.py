# File: 23_Stage2_TextCNN_DomURLBERT_All_Comm_Including_MessagePassing.py
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

print("="*100)
print("Stage 2: TextCNN (Email) + DomURLBERT (URL) - All Communication Styles + Message Passing")
print("="*100)

# ====================== CLEAN LABEL-MATCHED PAIRING ======================
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

print(f"Final samples with clean label-matched pairing: {len(df)}")

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
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, e, u):
        return self.fc(torch.cat([e, u], dim=1)).squeeze(-1)

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
        self.gate = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2), nn.Softmax(dim=1))
    def forward(self, e, u):
        comm = torch.cat([e, u], dim=1)
        gate = self.gate(comm)
        e_score = torch.sigmoid(torch.mean(e, dim=1))
        u_score = torch.sigmoid(torch.mean(u, dim=1))
        return gate[:, 0] * e_score + gate[:, 1] * u_score

class CrossAttention(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.fc = nn.Linear(dim*2, 1)
    def forward(self, e, u):
        Q = self.query(e)
        K = self.key(u)
        V = self.value(u)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2,-1)) / np.sqrt(dim), dim=-1)
        attended = torch.matmul(attn, V)
        comm = torch.cat([e, attended], dim=1)
        return self.fc(comm).squeeze(-1)

class MessagePassing(nn.Module):
    def __init__(self, dim=128, rounds=2):
        super().__init__()
        self.rounds = rounds
        self.update_e = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Dropout(0.3))
        self.update_u = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Dropout(0.3))
        self.fc = nn.Linear(dim*2, 1)
    def forward(self, email_feat, url_feat):
        e = email_feat
        u = url_feat
        for _ in range(self.rounds):
            e_new = self.update_e(torch.cat([e, u], dim=1))
            u_new = self.update_u(torch.cat([u, e], dim=1))
            e = e_new
            u = u_new
        comm = torch.cat([e, u], dim=1)
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
def train_style(comm_name, comm_class):
    print(f"\n=== Training {comm_name} ===")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"  Fold {fold+1}/10")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        email_model = TextCNN().to(device)
        url_model = DomURLBERT().to(device)
        comm_model = comm_class().to(device) if comm_class else None

        params = list(email_model.parameters()) + list(url_model.parameters())
        if comm_model: params += list(comm_model.parameters())

        optimizer = optim.AdamW(params, lr=8e-6, weight_decay=0.08)
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        max_epochs = 30

        for epoch in range(max_epochs):
            email_model.train()
            url_model.train()
            if comm_model: comm_model.train()

            train_loss = 0.0

            progress_bar = tqdm(range(0, len(train_df), 8), desc=f"Epoch {epoch+1:2d}/{max_epochs}", leave=False)

            for i in progress_bar:
                batch = train_df.iloc[i:i+8]
                labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

                e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

                e_feat = email_model(e_in.input_ids)
                u_feat = url_model(u_in.input_ids, u_in.attention_mask)

                if comm_name == "No Communication":
                    logits = torch.sigmoid(torch.mean(e_feat, dim=1)) * 0.5 + torch.sigmoid(torch.mean(u_feat, dim=1)) * 0.5
                    logits = logits * 2 - 1   # convert to logit scale
                elif comm_name == "Simple Concat":
                    logits = comm_model(e_feat, u_feat)
                elif comm_name == "Weighted Score":
                    e_score = torch.sigmoid(torch.mean(e_feat, dim=1))
                    u_score = torch.sigmoid(torch.mean(u_feat, dim=1))
                    logits = comm_model(e_score, u_score)
                elif comm_name in ["Gated Fusion", "Cross Attention"]:
                    logits = comm_model(e_feat, u_feat)
                elif comm_name == "Message Passing":
                    logits = comm_model(e_feat, u_feat)

                loss = criterion(logits, labels)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
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
        if comm_model: comm_model.eval()

        y_true = val_df['label'].values
        y_pred = []
        y_prob = []

        with torch.no_grad():
            for i in range(0, len(val_df), 32):
                batch = val_df.iloc[i:i+32]
                e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

                e_feat = email_model(e_in.input_ids)
                u_feat = url_model(u_in.input_ids, u_in.attention_mask)

                if comm_name == "No Communication":
                    e_score = torch.sigmoid(torch.mean(e_feat, dim=1))
                    u_score = torch.sigmoid(torch.mean(u_feat, dim=1))
                    prob = (e_score + u_score) / 2.0
                elif comm_name == "Simple Concat":
                    logits = comm_model(e_feat, u_feat)
                    prob = logits.sigmoid()
                elif comm_name == "Weighted Score":
                    e_score = torch.sigmoid(torch.mean(e_feat, dim=1))
                    u_score = torch.sigmoid(torch.mean(u_feat, dim=1))
                    prob = comm_model(e_score, u_score)
                else:
                    logits = comm_model(e_feat, u_feat)
                    prob = logits.sigmoid()

                pred = (prob > 0.5).cpu().numpy().astype(int)
                y_pred.extend(pred)
                y_prob.extend(prob.cpu().numpy())

        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        fold_results.append(metrics)

        del email_model, url_model
        if comm_model: del comm_model
        torch.cuda.empty_cache()
        gc.collect()

    avg_metrics = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
    print(f"\n=== {comm_name} Final Average ===")
    for k, v in avg_metrics.items():
        print(f"  {k.replace('_', ' ').title()}: {v:.4f}")
    return avg_metrics

# ====================== RUN ALL STYLES ======================
styles = {
    "No Communication": None,
    "Simple Concat": SimpleConcat,
    "Weighted Score": WeightedScore,
    "Gated Fusion": GatedFusion,
    "Cross Attention": CrossAttention,
    "Message Passing": MessagePassing
}

all_results = {}
for name, cls in styles.items():
    all_results[name] = train_style(name, cls)

print("\n=== FINAL COMPARISON TABLE ===")
comparison = pd.DataFrame(all_results).T
print(comparison[['accuracy', 'f1', 'roc_auc', 'mcc']])
comparison.to_csv("23_Stage2_All_Communication_Results.csv")
print("\nFull comparison saved to 23_Stage2_All_Communication_Results.csv")
