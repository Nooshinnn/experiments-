# File: 51_ZefangLiu_ZeroShot_Balanced_50_50.py
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print("=" * 140)
print("ZERO-SHOT GENERALIZATION ON ZEFANG-LIU DATASET")
print("Balanced 50% Phishing / 50% Legitimate Test Set")
print("Using your trained model: best_trained_model.pth")
print("=" * 140)

# ====================== YOUR MODEL ======================
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

email_model.eval()
url_model.eval()
comm_model.eval()

print("Loaded your trained model successfully.")

# ====================== LOAD ZEFANG-LIU DATASET ======================
print("Loading zefang-liu/phishing-email-dataset...")
dataset = load_dataset("zefang-liu/phishing-email-dataset", split="train")
df = pd.DataFrame(dataset)

df = df.rename(columns={"Email Text": "email_text", "Email Type": "label"})
df = df.dropna(subset=['email_text']).reset_index(drop=True)
df['label'] = df['label'].map({"Phishing Email": 1, "Safe Email": 0})
df = df[df['label'].isin([0, 1])].reset_index(drop=True)

def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else ""

df['url'] = df['email_text'].apply(extract_first_url)
df = df[df['url'] != ""].reset_index(drop=True)

print(f"Original dataset size: {len(df)} | Original phishing ratio: {df['label'].mean():.4f}")

# ====================== CREATE BALANCED 50/50 TEST SET ======================
# Separate phishing and legitimate
phishing = df[df['label'] == 1].copy()
legit = df[df['label'] == 0].copy()

# Take min size to make perfectly balanced
min_size = min(len(phishing), len(legit))
print(f"Creating balanced test set of {min_size * 2} samples (50% phishing / 50% legitimate)")

# Sample equally
phishing_sample = phishing.sample(n=min_size, random_state=42)
legit_sample = legit.sample(n=min_size, random_state=42)

balanced_df = pd.concat([phishing_sample, legit_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced test set size: {len(balanced_df)} | Phishing ratio: {balanced_df['label'].mean():.4f}")

test_df = balanced_df

# ====================== TOKENIZERS ======================
email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

# ====================== ZERO-SHOT EVALUATION ======================
def evaluate():
    y_true, y_prob = [], []
    with torch.no_grad():
        for i in range(0, len(test_df), 8):
            batch = test_df.iloc[i:i+8]
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

print("\n=== ZERO-SHOT RESULTS ON BALANCED ZEFANG-LIU (50/50) ===")
results = evaluate()
for k, v in results.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

pd.DataFrame([results]).to_csv("ZefangLiu_ZeroShot_Balanced_50_50.csv", index=False)
print("\nResults saved to ZefangLiu_ZeroShot_Balanced_50_50.csv")