# File: 59_5Fold_Original_Dataset_Full.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import warnings
from tqdm import tqdm
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import json

warnings.filterwarnings("ignore")
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*160)
print("5-FOLD CV - ORIGINAL DATASET (Baseline)")
print("="*160)

# ====================== MODEL ======================
class DistilBertEmail(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.adapter = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.5), nn.Linear(192, 128))
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.adapter(pooled)

class DomURLBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("amahdaouy/DomURLs_BERT")
        self.adapter = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.5), nn.Linear(192, 128))
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.adapter(pooled)

class MessagePassing(nn.Module):
    def __init__(self, dim=128, rounds=2):
        super().__init__()
        self.rounds = rounds
        self.update_e = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Dropout(0.5))
        self.update_u = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Dropout(0.5))
        self.fc = nn.Linear(dim*2, 1)
    def forward(self, e, u):
        for _ in range(self.rounds):
            e = self.update_e(torch.cat([e, u], dim=1))
            u = self.update_u(torch.cat([u, e], dim=1))
        return self.fc(torch.cat([e, u], dim=1)).squeeze(-1)

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

# Load Dataset
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset).rename(columns={"content": "email_text", "labels": "label"})
df = df[df['label'].isin([0, 1])].reset_index(drop=True)

def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else ""

df['url'] = df['email_text'].apply(extract_first_url)
df = df[df['url'] != ""].reset_index(drop=True)

X = df.reset_index(drop=True)
y = df['label'].values

# 5-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_fold_results = []
total_start = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n=== Fold {fold}/5 ===")
    train_df = X.iloc[train_idx].copy()
    val_df = X.iloc[val_idx].copy()

    email_model = DistilBertEmail().to(device)
    url_model = DomURLBERT().to(device)
    comm_model = MessagePassing().to(device)

    for param in email_model.model.parameters(): param.requires_grad = False
    for param in url_model.model.parameters(): param.requires_grad = False

    optimizer = optim.AdamW(list(email_model.adapter.parameters()) + list(url_model.adapter.parameters()) + list(comm_model.parameters()), lr=5e-6, weight_decay=0.12)
    pos_weight = torch.tensor((1 - train_df['label'].mean()) / train_df['label'].mean()).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(15):
        email_model.train(); url_model.train(); comm_model.train()
        total_loss = 0.0
        for i in tqdm(range(0, len(train_df), 8), desc=f"Fold {fold} Epoch {epoch+1}", leave=False):
            batch = train_df.iloc[i:i+8]
            labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

            emails = [str(t) for t in batch['email_text'].tolist()]
            urls = [str(u) for u in batch['url'].tolist()]

            e_in = email_tokenizer(emails, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(urls, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Final evaluation on validation set (add your evaluate function here)
    # For now placeholder
    predictions = np.random.randint(0, 2, len(val_df))
    probabilities = np.random.rand(len(val_df))
    true_labels = val_df['label'].values

    fold_result = {
        'fold': fold,
        'accuracy': accuracy_score(true_labels, predictions),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'mcc': matthews_corrcoef(true_labels, predictions),
        'roc_auc': roc_auc_score(true_labels, probabilities),
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist(),
        'true_labels': true_labels.tolist()
    }
    all_fold_results.append(fold_result)

# Save everything
with open('5fold_results_original.json', 'w') as f:
    json.dump(all_fold_results, f)

np.save('predictions_original.npy', np.concatenate([r['predictions'] for r in all_fold_results]))
np.save('probabilities_original.npy', np.concatenate([r['probabilities'] for r in all_fold_results]))
np.save('true_labels_original.npy', np.concatenate([r['true_labels'] for r in all_fold_results]))

print(f"Total Training Time: {(time.time() - total_start)/60:.2f} minutes")
print("All files saved successfully.")
