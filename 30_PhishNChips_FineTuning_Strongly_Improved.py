# File: 29_PhishNChips_FineTuning_Strongly_Improved_DistilBERT_DomURLBERT_MessagePassing.py
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

print("="*110)
print("STRONG FINE-TUNING ON PhishNChips")
print("Model: DistilBERT (Email) + DomURLBERT (URL) + Message Passing")
print("Heavy freezing + very low LR + weighted loss + early stopping on F1")
print("="*110)

# ====================== MODEL DEFINITION ======================
class DistilBertEmail(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(768, 128)
        self.dropout = nn.Dropout(0.4)
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(pooled))

class DomURLBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("amahdaouy/DomURLs_BERT")
        self.fc = nn.Linear(768, 128)
        self.dropout = nn.Dropout(0.4)
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(pooled))

class MessagePassing(nn.Module):
    def __init__(self, dim=128, rounds=2):
        super().__init__()
        self.rounds = rounds
        self.update_e = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Dropout(0.4))
        self.update_u = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Dropout(0.4))
        self.fc = nn.Linear(dim*2, 1)
    def forward(self, e, u):
        for _ in range(self.rounds):
            e = self.update_e(torch.cat([e, u], dim=1))
            u = self.update_u(torch.cat([u, e], dim=1))
        return self.fc(torch.cat([e, u], dim=1)).squeeze(-1)

# ====================== LOAD PhishNChips ======================
print("\nLoading PhishNChips dataset...")
dataset = load_dataset("AreLit/PhishNChips", "emails", split="core")
df = pd.DataFrame(dataset)

df = df.rename(columns={
    "email_content": "email_text",
    "phish_label": "label",
    "url_raw": "url"
})

df = df[df['label'].isin([0, 1])].reset_index(drop=True)
print(f"Total samples: {len(df)} | Positive (phishing) ratio: {df['label'].mean():.4f}")

# Split into train/val
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

# ====================== TOKENIZERS ======================
email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

# ====================== MODELS ======================
email_model = DistilBertEmail().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing().to(device)

# ====================== WEIGHTED LOSS ======================
pos_ratio = df['label'].mean()
pos_weight = torch.tensor((1 - pos_ratio) / (pos_ratio + 1e-8)).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ====================== OPTIMIZER ======================
optimizer = optim.AdamW(
    list(email_model.parameters()) +
    list(url_model.parameters()) +
    list(comm_model.parameters()),
    lr=2e-6,
    weight_decay=0.05
)

# ====================== TRAINING PARAMETERS ======================
batch_size = 6
max_epochs = 10
patience = 4
best_val_f1 = 0.0
patience_counter = 0
accumulation_steps = 4

print("\nStarting training...\n")

for epoch in range(max_epochs):
    # Train phase
    email_model.train()
    url_model.train()
    comm_model.train()

    # Heavy freezing: freeze base transformers for first 3 epochs
    freeze_base = (epoch < 3)
    for param in email_model.model.parameters():
        param.requires_grad = not freeze_base
    for param in url_model.model.parameters():
        param.requires_grad = not freeze_base

    train_loss = 0.0
    optimizer.zero_grad()

    progress = tqdm(range(0, len(train_df), batch_size), desc=f"Epoch {epoch+1}/{max_epochs} [Train]")

    for step, i in enumerate(progress):
        batch = train_df.iloc[i:i + batch_size]
        labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

        e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)

        logits = comm_model(e_feat, u_feat)
        loss = criterion(logits, labels) / accumulation_steps

        loss.backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == (len(train_df) // batch_size):
            torch.nn.utils.clip_grad_norm_(
                list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item() * accumulation_steps
        progress.set_postfix({'Loss': f'{train_loss / (step + 1):.4f}'})

    # Validation phase
    email_model.eval()
    url_model.eval()
    comm_model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for i in range(0, len(val_df), batch_size):
            batch = val_df.iloc[i:i + batch_size]
            e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)

            logits = comm_model(e_feat, u_feat)
            prob = logits.sigmoid().cpu().numpy()
            y_true.extend(batch['label'].values)
            y_prob.extend(prob)

    val_f1 = f1_score(y_true, np.array(y_prob) > 0.5, zero_division=0)
    print(f"Epoch {epoch+1} | Val F1: {val_f1:.4f}")

    # Early stopping based on F1
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        print("    → New best validation F1")
    else:
        patience_counter += 1
        print(f"    → No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

# ====================== FINAL EVALUATION ON FULL DATASET ======================
print("\n=== FINAL RESULTS AFTER FINE-TUNING ===")
email_model.eval()
url_model.eval()
comm_model.eval()

y_true = []
y_pred = []
y_prob = []

with torch.no_grad():
    for i in range(0, len(df), 16):
        batch = df.iloc[i:i+16]
        e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)

        logits = comm_model(e_feat, u_feat)
        prob = logits.sigmoid().cpu().numpy()
        pred = (prob > 0.5).astype(int)

        y_true.extend(batch['label'].values)
        y_pred.extend(pred)
        y_prob.extend(prob)

metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, average='binary', zero_division=0),
    "recall": recall_score(y_true, y_pred, average='binary', zero_division=0),
    "f1": f1_score(y_true, y_pred, average='binary', zero_division=0),
    "mcc": matthews_corrcoef(y_true, y_pred),
    "roc_auc": roc_auc_score(y_true, y_prob),
    "avg_precision": average_precision_score(y_true, y_prob),
}

print("\nFinal Results on PhishNChips:")
for k, v in metrics.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

pd.DataFrame([metrics]).to_csv("PhishNChips_FineTuned_Strongly_Improved.csv", index=False)
print("\nResults saved to: PhishNChips_FineTuned_Strongly_Improved.csv")
