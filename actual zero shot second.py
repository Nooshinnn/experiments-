# File: 53_PhishNChips_Better_DomainShift_Experiment.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import warnings
from tqdm import tqdm
import numpy as np
import re

warnings.filterwarnings("ignore")
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print("="*160)
print("BETTER DOMAIN SHIFT EXPERIMENT ON PHISHNCHIPS")
print("Zero-shot → Partial Unfreezing + Scheduler → Final Test")
print("Using your trained model: best_trained_model.pth")
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

# Load your trained model
checkpoint = torch.load("best_trained_model.pth", map_location=device)
email_model = DistilBertEmail().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing().to(device)
email_model.load_state_dict(checkpoint['email_model'])
url_model.load_state_dict(checkpoint['url_model'])
comm_model.load_state_dict(checkpoint['comm_model'])

print("Loaded your trained model successfully.")

# ====================== LOAD PHISHNCHIPS ======================
dataset = load_dataset("AreLit/PhishNChips", "emails", split="core")
df = pd.DataFrame(dataset)
df = df.rename(columns={"email_content": "email_text", "phish_label": "label", "url_raw": "url"})
df = df[df['label'].isin([0, 1])].reset_index(drop=True)
df = df.dropna(subset=['email_text']).reset_index(drop=True)

def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else ""

df['url'] = df['url'].fillna("").apply(lambda x: str(x) if x else extract_first_url(str(x)))
df = df[df['url'] != ""].reset_index(drop=True)

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

# ====================== EVALUATION ======================
def evaluate(df_eval):
    y_true, y_prob = [], []
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i+8]
            emails = [str(t) for t in batch['email_text'].tolist()]
            urls = [str(u) for u in batch['url'].tolist()]

            e_in = email_tokenizer(emails, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(urls, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            prob = logits.sigmoid().cpu().numpy()
            y_true.extend(batch['label'].values)
            y_prob.extend(prob)

    y_pred = (np.array(y_prob) > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

# ====================== STAGE 1: ZERO-SHOT ======================
print("\n=== STAGE 1: ZERO-SHOT (Domain Shift Impact) ===")
zero_shot = evaluate(test_df)
for k, v in zero_shot.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

# ====================== STAGE 2: BETTER DOMAIN ADAPTATION (Partial Unfreezing) ======================
print("\n=== STAGE 2: BETTER DOMAIN ADAPTATION - Partial Unfreezing ===")

# Unfreeze last 2 layers of both transformers + adapters + comm
for param in list(email_model.model.parameters())[-4:]:   # last 2 layers
    param.requires_grad = True
for param in list(url_model.model.parameters())[-4:]:
    param.requires_grad = True

optimizer = optim.AdamW(
    [p for p in email_model.parameters() if p.requires_grad] +
    [p for p in url_model.parameters() if p.requires_grad] +
    list(comm_model.parameters()),
    lr=8e-6, weight_decay=0.12
)

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

pos_weight = torch.tensor((1 - train_df['label'].mean()) / train_df['label'].mean()).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_f1 = 0.0
patience = 3
counter = 0

for epoch in range(12):
    email_model.train()
    url_model.train()
    comm_model.train()
    total_loss = 0.0

    for i in tqdm(range(0, len(train_df), 8), desc=f"Epoch {epoch+1}"):
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

    print(f"Epoch {epoch+1} | Loss: {total_loss / (len(train_df)//8 + 1):.4f}")

    val_results = evaluate(val_df)
    print(f"Val F1: {val_results['f1']:.4f}")

    scheduler.step(val_results['f1'])

    if val_results['f1'] > best_f1 + 0.005:
        best_f1 = val_results['f1']
        counter = 0
        torch.save({
            'email_model': email_model.state_dict(),
            'url_model': url_model.state_dict(),
            'comm_model': comm_model.state_dict(),
        }, "phishnchips_better_domain_adapted_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

# ====================== STAGE 3: FINAL TEST ======================
print("\n=== STAGE 3: FINAL TEST AFTER BETTER DOMAIN ADAPTATION ===")
final_results = evaluate(test_df)
for k, v in final_results.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

print("\nBetter domain-adapted model saved as 'phishnchips_better_domain_adapted_model.pth'")