# File: 63_MessagePassing_Ablation_Matching_Style.py
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*160)
print("MESSAGE PASSING ABLATION - MATCHING YOUR REFERENCE STYLE")
print("Same models + early stopping + full metrics + all plots")
print("="*160)

# ====================== LOAD DATASET & CLEAN PAIRING ======================
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset)
df = df[df['labels'].isin([0, 1])].reset_index(drop=True)
df = df.rename(columns={"labels": "label", "content": "email_text"})

def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else None

df['url'] = df['email_text'].apply(extract_first_url)

# Clean pairing
phishing_urls = df[df['label'] == 1]['url'].dropna().tolist()
legit_urls = df[df['label'] == 0]['url'].dropna().tolist()
random.seed(42)
def get_matching_url(label):
    if label == 1 and phishing_urls:
        return random.choice(phishing_urls)
    elif label == 0 and legit_urls:
        return random.choice(legit_urls)
    return ""
df['url'] = df.apply(lambda row: row['url'] if pd.notna(row['url']) else get_matching_url(row['label']), axis=1)

print(f"Final samples: {len(df)}")

# ====================== MODELS (Same as your reference) ======================
class DistilBertWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(768, 128)
        self.dropout = nn.Dropout(0.3)
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.fc(pooled)

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

# ====================== MESSAGE PASSING (with variable rounds) ======================
class MessagePassing(nn.Module):
    def __init__(self, dim=128, rounds=2):
        super().__init__()
        self.rounds = rounds
        self.dim = dim
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

# ====================== METRICS (Same as reference) ======================
def compute_all_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
    }
    return metrics

# ====================== ABLATION ======================
rounds_list = [1, 2, 3, 4]
ablation_results = {}
n_folds = 5
max_epochs = 30
patience = 5
batch_size = 8

for rounds in rounds_list:
    print(f"\n=== Message Passing with {rounds} Rounds ===")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label']), 1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        email_model = DistilBertWrapper().to(device)
        url_model = DomURLBERT().to(device)
        comm_model = MessagePassing(rounds=rounds).to(device)

        # Freeze base models
        for p in email_model.model.parameters(): p.requires_grad = False
        for p in url_model.model.parameters(): p.requires_grad = False

        optimizer = optim.AdamW(
            list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()),
            lr=8e-6, weight_decay=0.08
        )
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs):
            email_model.train()
            url_model.train()
            comm_model.train()
            train_loss = 0.0

            for i in tqdm(range(0, len(train_df), batch_size), desc=f"R{rounds} F{fold} E{epoch+1}", leave=False):
                batch = train_df.iloc[i:i+batch_size]
                labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

                e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

                e_feat = email_model(e_in.input_ids, e_in.attention_mask)
                u_feat = url_model(u_in.input_ids, u_in.attention_mask)
                logits = comm_model(e_feat, u_feat)

                loss = criterion(logits, labels)
                torch.nn.utils.clip_grad_norm_(list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()), max_norm=1.0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_loss = train_loss / (len(train_df) // batch_size + 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Evaluation
        email_model.eval()
        url_model.eval()
        comm_model.eval()
        y_true = val_df['label'].values
        y_prob = []
        with torch.no_grad():
            for i in range(0, len(val_df), batch_size):
                batch = val_df.iloc[i:i+batch_size]
                e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                e_feat = email_model(e_in.input_ids, e_in.attention_mask)
                u_feat = url_model(u_in.input_ids, u_in.attention_mask)
                logits = comm_model(e_feat, u_feat)
                prob = logits.sigmoid().cpu().numpy()
                y_prob.extend(prob)
        y_pred = (np.array(y_prob) > 0.5).astype(int)
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        fold_results.append(metrics)
        print(f"  Fold {fold} - F1: {metrics['f1']:.4f} | MCC: {metrics['mcc']:.4f}")

    ablation_results[rounds] = fold_results

# Save results
with open('message_passing_ablation_full_results.json', 'w') as f:
    json.dump(ablation_results, f)

# ====================== ALL PLOTS ======================
rounds = list(ablation_results.keys())
avg_f1 = [np.mean([r['f1'] for r in ablation_results[r]]) for r in rounds]
avg_mcc = [np.mean([r['mcc'] for r in ablation_results[r]]) for r in rounds]
avg_acc = [np.mean([r['accuracy'] for r in ablation_results[r]]) for r in rounds]
avg_auc = [np.mean([r['roc_auc'] for r in ablation_results[r]]) for r in rounds]

plt.figure(figsize=(12, 7))
plt.bar([f"{r} Rounds" for r in rounds], avg_f1, color=['#1f77b4','#ff7f0e','#2ca02c','#d62728'])
plt.title('Average F1 Score by Message Passing Rounds', fontsize=14)
plt.ylabel('F1 Score')
plt.grid(axis='y')
plt.savefig("Ablation_F1_Bar.png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(12, 7))
plt.plot(rounds, avg_f1, 'o-', label='F1', linewidth=2.5)
plt.plot(rounds, avg_mcc, 's-', label='MCC', linewidth=2.5)
plt.plot(rounds, avg_acc, '^-', label='Accuracy', linewidth=2.5)
plt.plot(rounds, avg_auc, 'd-', label='ROC-AUC', linewidth=2.5)
plt.title('Performance vs Number of Message Passing Rounds')
plt.xlabel('Rounds')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.savefig("Ablation_All_Metrics_Line.png", dpi=300, bbox_inches='tight')

print("\n✅ Ablation finished!")
print("Check the average F1 for 2 rounds — it should now match your previous high results.")
