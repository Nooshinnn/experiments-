# File: 34_zefang-liu_phishing_ZeroShot.py
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import warnings
import numpy as np
import re

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("="*120)
print("ZERO-SHOT EVALUATION ON zefang-liu/phishing-email-dataset")
print("Model: DistilBERT (Email) + DomURLBERT (URL) + Message Passing")
print("No fine-tuning - Pure generalization test")
print("="*120)

# ====================== MODEL (Same as your best) ======================
class DistilBertEmail(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.adapter = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.5), nn.Linear(192, 128))
        for param in self.model.parameters(): param.requires_grad = False
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.adapter(pooled)

class DomURLBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("amahdaouy/DomURLs_BERT")
        self.adapter = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.5), nn.Linear(192, 128))
        for param in self.model.parameters(): param.requires_grad = False
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
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

# ====================== LOAD DATASET ======================
print("\nLoading zefang-liu/phishing-email-dataset...")
dataset = load_dataset("zefang-liu/phishing-email-dataset", split="train")
df = pd.DataFrame(dataset)

df = df.rename(columns={"Email Text": "email_text", "Email Type": "label"})

# Convert labels
label_map = {"Safe Email": 0, "Phishing Email": 1}
df['label'] = df['label'].map(label_map)

df = df.dropna(subset=['email_text'])
df['email_text'] = df['email_text'].astype(str)

def extract_first_url(text):
    urls = re.findall(r'https?://\S+', text)
    return urls[0] if urls else ""

df['url'] = df['email_text'].apply(extract_first_url)

df = df[df['label'].isin([0, 1])].reset_index(drop=True)
print(f"Total clean samples: {len(df)} | Phishing ratio: {df['label'].mean():.4f}")

# Strict split
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ====================== TOKENIZERS & MODELS ======================
email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

email_model = DistilBertEmail().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing().to(device)

email_model.eval()
url_model.eval()
comm_model.eval()

# ====================== ZERO-SHOT EVALUATION ON TEST SET ======================
print("\nRunning Zero-Shot Evaluation on Held-out Test Set...")

y_true = []
y_prob = []

with torch.no_grad():
    for i in range(0, len(test_df), 16):
        batch = test_df.iloc[i:i+16]
        email_texts = [str(text) for text in batch['email_text'].tolist()]
        urls = [str(url) for url in batch['url'].tolist()]

        e_in = email_tokenizer(email_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(urls, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)

        logits = comm_model(e_feat, u_feat)
        prob = logits.sigmoid().cpu().numpy()
        y_true.extend(batch['label'].values)
        y_prob.extend(prob)

y_pred = (np.array(y_prob) > 0.5).astype(int)

# ====================== RESULTS ======================
final_metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    "f1": f1_score(y_true, y_pred, zero_division=0),
    "mcc": matthews_corrcoef(y_true, y_pred),
    "roc_auc": roc_auc_score(y_true, y_prob),
}

print("\n=== ZERO-SHOT RESULTS ON HELD-OUT TEST SET ===")
for k, v in final_metrics.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Legitimate (0)', 'Phishing (1)'], digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

pd.DataFrame([final_metrics]).to_csv("zefang_liu_phishing_ZeroShot_Results.csv", index=False)
print("\nZero-shot results saved to zefang_liu_phishing_ZeroShot_Results.csv")
