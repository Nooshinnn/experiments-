# File: Stage2_OneStyle_TextCNN_DomURLBERT.py
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

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================== CONFIG - CHANGE ONLY THIS ======================
communication_name = "Simple Concat"   # <<< CHANGE THIS LINE ONLY >>>
# Valid options: 
# "No Communication", "Simple Concat", "Weighted Score", "Gated Fusion", 
# "Cross Attention", "Message Passing"

n_folds = 5
max_epochs = 30
patience = 5
batch_size = 8

print(f"Running Style: {communication_name} | {n_folds}-fold | Max epochs: {max_epochs}")

# ====================== LOAD DATASET & CLEAN PAIRING ======================
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

print(f"Final samples with clean pairing: {len(df)}")

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
    def forward(self, e, u):
        for _ in range(self.rounds):
            e_new = self.update_e(torch.cat([e, u], dim=1))
            u_new = self.update_u(torch.cat([u, e], dim=1))
            e, u = e_new, u_new
        comm = torch.cat([e, u], dim=1)
        return self.fc(comm).squeeze(-1)

# No Communication baseline
class NoCommunication:
    @staticmethod
    def forward(e_feat, u_feat):
        e_score = torch.sigmoid(torch.mean(e_feat, dim=1))
        u_score = torch.sigmoid(torch.mean(u_feat, dim=1))
        return (e_score + u_score) / 2.0

# ====================== SELECT COMMUNICATION MODULE ======================
comm_dict = {
    "No Communication": None,
    "Simple Concat": SimpleConcat,
    "Weighted Score": WeightedScore,
    "Gated Fusion": GatedFusion,
    "Cross Attention": CrossAttention,
    "Message Passing": MessagePassing
}

comm_class = comm_dict[communication_name]

# ====================== METRICS FUNCTION ======================
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

# ====================== TRAINING LOOP ======================
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"\n--- Fold {fold+1}/{n_folds} ---")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    email_model = TextCNN().to(device)
    url_model = DomURLBERT().to(device)
    comm_model = comm_class().to(device) if comm_class is not None else None

    params = list(email_model.parameters()) + list(url_model.parameters())
    if comm_model:
        params += list(comm_model.parameters())

    optimizer = optim.AdamW(params, lr=8e-6, weight_decay=0.08)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        email_model.train()
        url_model.train()
        if comm_model: comm_model.train()

        train_loss = 0.0

        progress_bar = tqdm(range(0, len(train_df), batch_size), desc=f"Epoch {epoch+1:2d}/{max_epochs}", leave=False)

        for i in progress_bar:
            batch = train_df.iloc[i:i+batch_size]
            labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

            e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

            e_feat = email_model(e_in.input_ids)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)

            # Forward pass based on style
            if communication_name == "No Communication":
                prob = NoCommunication.forward(e_feat, u_feat)
                logits = prob * 2 - 1   # convert to logit scale for loss
            elif communication_name == "Simple Concat":
                logits = comm_model(e_feat, u_feat)
            elif communication_name == "Weighted Score":
                e_score = torch.sigmoid(torch.mean(e_feat, dim=1))
                u_score = torch.sigmoid(torch.mean(u_feat, dim=1))
                logits = comm_model(e_score, u_score)
            else:
                logits = comm_model(e_feat, u_feat)

            loss = criterion(logits, labels)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'Train Loss': f'{train_loss / (i//batch_size + 1):.4f}'})

        avg_train_loss = train_loss / (len(train_df) // batch_size + 1)

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # ====================== EVALUATION ======================
    email_model.eval()
    url_model.eval()
    if comm_model: comm_model.eval()

    y_true = val_df['label'].values
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for i in range(0, len(val_df), batch_size):
            batch = val_df.iloc[i:i+batch_size]
            e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

            e_feat = email_model(e_in.input_ids)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)

            if communication_name == "No Communication":
                prob = NoCommunication.forward(e_feat, u_feat).cpu().numpy()
            elif communication_name == "Simple Concat":
                logits = comm_model(e_feat, u_feat)
                prob = logits.sigmoid().cpu().numpy()
            elif communication_name == "Weighted Score":
                e_score = torch.sigmoid(torch.mean(e_feat, dim=1))
                u_score = torch.sigmoid(torch.mean(u_feat, dim=1))
                prob = comm_model(e_score, u_score).cpu().numpy()
            else:
                logits = comm_model(e_feat, u_feat)
                prob = logits.sigmoid().cpu().numpy()

            pred = (prob > 0.5).astype(int)
            y_pred.extend(pred)
            y_prob.extend(prob)

    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    fold_results.append(metrics)

    del email_model, url_model
    if comm_model: del comm_model
    torch.cuda.empty_cache()
    gc.collect()

# ====================== FINAL RESULTS ======================
avg_metrics = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
print(f"\n=== {communication_name} Final Results ({n_folds}-fold) ===")
for k, v in avg_metrics.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

result_df = pd.DataFrame([avg_metrics])
result_df.to_csv(f"Stage2_{communication_name.replace(' ', '_')}_results.csv", index=False)
print(f"\nResults saved to Stage2_{communication_name.replace(' ', '_')}_results.csv")
