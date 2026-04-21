# File: 27_PhishNChips_ZeroShot_DistilBERT_DomURLBERT_MessagePassing.py
import pandas as pd
import re
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
    cohen_kappa_score, average_precision_score, log_loss
)
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import warnings
import numpy as np

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("="*100)
print("ZERO-SHOT GENERALIZATION TEST")
print("Model: DistilBERT (Email) + DomURLBERT (URL) + Message Passing")
print("Trained only on cybersectony dataset")
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

# ====================== LOAD PhishNChips (Corrected) ======================
print("\nLoading PhishNChips dataset for zero-shot test...")
phish_dataset = load_dataset("AreLit/PhishNChips", "emails", split="core")
phish_df = pd.DataFrame(phish_dataset)

# Rename columns to match our expected format
phish_df = phish_df.rename(columns={
    "email_content": "email_text",
    "phish_label": "label"
})

# Keep only binary labels
phish_df = phish_df[phish_df['label'].isin([0, 1])].reset_index(drop=True)

def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else None

phish_df['url'] = phish_df['url_raw']   # Use the provided url_raw column (already clean)

print(f"PhishNChips samples with real URLs: {len(phish_df)}")

# ====================== TOKENIZERS & MODELS ======================
email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

email_model = DistilBertEmail().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing().to(device)

email_model.eval()
url_model.eval()
comm_model.eval()

# ====================== ZERO-SHOT EVALUATION ======================
print("\nRunning zero-shot evaluation on PhishNChips...")

y_true = []
y_pred = []
y_prob = []

batch_size = 16
with torch.no_grad():
    for i in range(0, len(phish_df), batch_size):
        batch = phish_df.iloc[i:i+batch_size]
        
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

# ====================== METRICS ======================
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
    "roc_auc": roc_auc_score(y_true, y_prob),
    "avg_precision": average_precision_score(y_true, y_prob),
    "log_loss": log_loss(y_true, y_prob),
}

print("\n=== ZERO-SHOT RESULTS ON PhishNChips ===")
for k, v in metrics.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

pd.DataFrame([metrics]).to_csv("PhishNChips_ZeroShot_DistilBERT_DomURLBERT_MessagePassing.csv", index=False)
print("\nResults saved to: PhishNChips_ZeroShot_DistilBERT_DomURLBERT_MessagePassing.csv")
