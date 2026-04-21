# File: 29_PhishNChips_Safe_Generalization_Test.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import warnings
from tqdm import tqdm
import numpy as np

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("="*120)
print("SAFE GENERALIZATION TEST ON PhishNChips")
print("DistilBERT (Email) + DomURLBERT (URL) + Message Passing")
print("Strict train/val/test split • Zero-shot vs Fine-tuned comparison")
print("="*120)

# ====================== MODEL ======================
class DistilBertEmail(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(768, 128)
        self.dropout = nn.Dropout(0.35)
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(pooled))

class DomURLBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("amahdaouy/DomURLs_BERT")
        self.fc = nn.Linear(768, 128)
        self.dropout = nn.Dropout(0.35)
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(pooled))

class MessagePassing(nn.Module):
    def __init__(self, dim=128, rounds=2):
        super().__init__()
        self.rounds = rounds
        self.update_e = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Dropout(0.35))
        self.update_u = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Dropout(0.35))
        self.fc = nn.Linear(dim*2, 1)
    def forward(self, e, u):
        for _ in range(self.rounds):
            e = self.update_e(torch.cat([e, u], dim=1))
            u = self.update_u(torch.cat([u, e], dim=1))
        return self.fc(torch.cat([e, u], dim=1)).squeeze(-1)

# ====================== LOAD & STRICT SPLIT ======================
print("\nLoading PhishNChips dataset...")
dataset = load_dataset("AreLit/PhishNChips", "emails", split="core")
df = pd.DataFrame(dataset)

df = df.rename(columns={
    "email_content": "email_text",
    "phish_label": "label",
    "url_raw": "url"
})

df = df[df['label'].isin([0, 1])].reset_index(drop=True)
print(f"Total samples: {len(df)} | Phishing ratio: {df['label'].mean():.4f}")

# Strict 70% train / 15% val / 15% test split
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ====================== TOKENIZERS ======================
email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

# ====================== MODELS ======================
email_model = DistilBertEmail().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing().to(device)

# ====================== ZERO-SHOT ON TEST SET ======================
print("\n=== ZERO-SHOT EVALUATION ON HELD-OUT TEST SET ===")
email_model.eval()
url_model.eval()
comm_model.eval()

y_true, y_prob = [], []

with torch.no_grad():
    for i in range(0, len(test_df), 16):
        batch = test_df.iloc[i:i+16]
        e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)

        logits = comm_model(e_feat, u_feat)
        prob = logits.sigmoid().cpu().numpy()
        y_true.extend(batch['label'].values)
        y_prob.extend(prob)

zero_shot_metrics = {
    "accuracy": accuracy_score(y_true, np.array(y_prob) > 0.5),
    "f1": f1_score(y_true, np.array(y_prob) > 0.5, zero_division=0),
    "mcc": matthews_corrcoef(y_true, np.array(y_prob) > 0.5),
    "roc_auc": roc_auc_score(y_true, y_prob),
}

print("Zero-shot on Test Set:")
for k, v in zero_shot_metrics.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

# ====================== FINE-TUNING ON TRAIN SPLIT ======================
print("\n=== FINE-TUNING ON TRAIN SPLIT (with validation monitoring) ===")

pos_weight = torch.tensor((1 - train_df['label'].mean()) / train_df['label'].mean()).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(
    list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()),
    lr=1e-6, weight_decay=0.05
)

batch_size = 6
max_epochs = 8
patience = 3
best_val_f1 = 0.0
patience_counter = 0

for epoch in range(max_epochs):
    email_model.train()
    url_model.train()
    comm_model.train()

    # Freeze base layers for first 3 epochs
    if epoch < 3:
        for p in email_model.model.parameters(): p.requires_grad = False
        for p in url_model.model.parameters(): p.requires_grad = False

    train_loss = 0.0
    progress = tqdm(range(0, len(train_df), batch_size), desc=f"Epoch {epoch+1}")

    for i in progress:
        batch = train_df.iloc[i:i+batch_size]
        labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

        e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)

        logits = comm_model(e_feat, u_feat)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress.set_postfix({'Loss': f'{train_loss/(i//batch_size+1):.4f}'})

    # Validation
    email_model.eval()
    url_model.eval()
    comm_model.eval()
    y_true_val, y_prob_val = [], []

    with torch.no_grad():
        for i in range(0, len(val_df), batch_size):
            batch = val_df.iloc[i:i+batch_size]
            e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)

            logits = comm_model(e_feat, u_feat)
            prob = logits.sigmoid().cpu().numpy()
            y_true_val.extend(batch['label'].values)
            y_prob_val.extend(prob)

    val_f1 = f1_score(y_true_val, np.array(y_prob_val) > 0.5, zero_division=0)
    print(f"Epoch {epoch+1} | Val F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break

# ====================== FINAL TEST EVALUATION ======================
print("\n=== FINAL TEST SET EVALUATION (True Generalization) ===")
email_model.eval()
url_model.eval()
comm_model.eval()

y_true, y_prob = [], []

with torch.no_grad():
    for i in range(0, len(test_df), 16):
        batch = test_df.iloc[i:i+16]
        e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)

        logits = comm_model(e_feat, u_feat)
        prob = logits.sigmoid().cpu().numpy()
        y_true.extend(batch['label'].values)
        y_prob.extend(prob)

final_metrics = {
    "accuracy": accuracy_score(y_true, np.array(y_prob) > 0.5),
    "balanced_accuracy": balanced_accuracy_score(y_true, np.array(y_prob) > 0.5),
    "f1": f1_score(y_true, np.array(y_prob) > 0.5, zero_division=0),
    "mcc": matthews_corrcoef(y_true, np.array(y_prob) > 0.5),
    "roc_auc": roc_auc_score(y_true, y_prob),
}

print("\nTest Set Results (True Held-out):")
for k, v in final_metrics.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

pd.DataFrame([final_metrics]).to_csv("PhishNChips_Safe_Test_Results.csv", index=False)
print("\nSafe results saved to PhishNChips_Safe_Test_Results.csv")
