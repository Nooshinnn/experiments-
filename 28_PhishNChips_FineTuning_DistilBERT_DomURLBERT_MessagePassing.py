# File: 28_PhishNChips_FineTuning_DistilBERT_DomURLBERT_MessagePassing.py
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
    cohen_kappa_score, average_precision_score, log_loss
)
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("="*100)
print("FINE-TUNING ON PhishNChips (5 epochs)")
print("Model: DistilBERT (Email) + DomURLBERT (URL) + Message Passing")
print("="*100)

# ====================== BEST MODEL ======================
class DistilBertEmail(nn.Module):
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

# ====================== LOAD PhishNChips ======================
phish_df = load_dataset("AreLit/PhishNChips", split="core_emails")
phish_df = pd.DataFrame(phish_df)

phish_df = phish_df[phish_df['label'].isin([0, 1])].reset_index(drop=True)
phish_df = phish_df.rename(columns={"content": "email_text"})

def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else None

phish_df['url'] = phish_df['email_text'].apply(extract_first_url)
phish_df = phish_df[phish_df['url'].notna()].reset_index(drop=True)

print(f"PhishNChips samples for fine-tuning: {len(phish_df)}")

# ====================== TOKENIZERS & MODELS ======================
email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

email_model = DistilBertEmail().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing().to(device)

optimizer = optim.AdamW(
    list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()),
    lr=5e-5, weight_decay=0.01
)
criterion = nn.BCEWithLogitsLoss()

# ====================== FINE-TUNING ======================
print("\nStarting fine-tuning (5 epochs)...")
batch_size = 8
for epoch in range(5):
    email_model.train()
    url_model.train()
    comm_model.train()
    train_loss = 0.0

    progress = tqdm(range(0, len(phish_df), batch_size), desc=f"Epoch {epoch+1}/5")
    for i in progress:
        batch = phish_df.iloc[i:i+batch_size]
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

    print(f"Epoch {epoch+1} completed. Avg Loss: {train_loss / (len(phish_df)//batch_size + 1):.4f}")

# ====================== FINAL EVALUATION ======================
print("\nEvaluating after fine-tuning...")
email_model.eval()
url_model.eval()
comm_model.eval()

y_true = []
y_pred = []
y_prob = []

with torch.no_grad():
    for i in range(0, len(phish_df), 16):
        batch = phish_df.iloc[i:i+16]
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

print("\n=== RESULTS AFTER FINE-TUNING ON PhishNChips ===")
for k, v in metrics.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

pd.DataFrame([metrics]).to_csv("PhishNChips_FineTuned_DistilBERT_DomURLBERT_MessagePassing.csv", index=False)
print("\nResults saved to: PhishNChips_FineTuned_DistilBERT_DomURLBERT_MessagePassing.csv")
