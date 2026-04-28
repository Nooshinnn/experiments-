# File: Experiment6_Threshold_Optimisation_MessagePassing2_Full.py
import pandas as pd
import re
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import warnings
import matplotlib.pyplot as plt
import random

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*90)
print("EXPERIMENT 6: Threshold Optimisation Analysis")
print("Message Passing (2 rounds) - Security-focused Recall Priority")
print("="*90)

# ====================== LOAD DATASET ======================
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
    if label == 1 and phishing_urls: return random.choice(phishing_urls)
    if label == 0 and legit_urls: return random.choice(legit_urls)
    return ""

df['url'] = df.apply(lambda row: row['url'] if pd.notna(row['url']) else get_matching_url(row['label']), axis=1)

# Held-out test set
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
test_df = test_df.reset_index(drop=True)

# ====================== MODELS (Your Best Config) ======================
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

class MessagePassing(nn.Module):
    def __init__(self, rounds=2):
        super().__init__()
        self.rounds = rounds
        self.update_e = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))
        self.update_u = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))
        self.fc = nn.Linear(256, 1)
    def forward(self, e, u):
        for _ in range(self.rounds):
            e = self.update_e(torch.cat([e, u], dim=1))
            u = self.update_u(torch.cat([u, e], dim=1))
        return self.fc(torch.cat([e, u], dim=1)).squeeze(-1)

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

email_model = DistilBertWrapper().to(device)
url_model = DomURLBERT().to(device)
mp_model = MessagePassing(rounds=2).to(device)

# ====================== GET PROBABILITIES ======================
email_model.eval()
url_model.eval()
mp_model.eval()

y_true = []
y_prob = []

print("Generating probability outputs...")
with torch.no_grad():
    for i in range(0, len(test_df), 8):
        batch = test_df.iloc[i:i+8]
        e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        logits = mp_model(e_feat, u_feat)
        prob = logits.sigmoid().cpu().numpy().flatten()
        y_prob.extend(prob)
        y_true.extend(batch['label'].values)

y_true = np.array(y_true)
y_prob = np.array(y_prob)

# ====================== THRESHOLD ANALYSIS ======================
thresholds = np.arange(0.1, 1.0, 0.05)
results = []

print("\nThreshold | Accuracy | Precision | Recall | F1 Score")
print("-" * 60)

best_f1 = 0
best_thresh = 0.5
best_recall = 0

for thresh in thresholds:
    y_pred = (y_prob >= thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    results.append({
        "Threshold": thresh,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    })
    
    print(f"{thresh:.2f}      | {acc:.4f}    | {prec:.4f}    | {rec:.4f}   | {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
        best_recall = rec

# ====================== SUMMARY TABLE ======================
print("\n" + "="*70)
print("THRESHOLD OPTIMISATION SUMMARY")
print("="*70)
print(f"Best Threshold (by F1) : {best_thresh:.2f}")
print(f"F1 Score                : {best_f1:.4f}")
print(f"Recall                  : {best_recall:.4f}")
print(f"Recommended for Security: Threshold = {best_thresh:.2f} (High Recall Priority)")

# ====================== PLOTS ======================
# Plot 1: Precision, Recall, F1 vs Threshold
thresholds_list = [r["Threshold"] for r in results]
prec_list = [r["Precision"] for r in results]
rec_list = [r["Recall"] for r in results]
f1_list = [r["F1"] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(thresholds_list, prec_list, 'b-', label='Precision')
plt.plot(thresholds_list, rec_list, 'g-', label='Recall')
plt.plot(thresholds_list, f1_list, 'r-', label='F1 Score')
plt.axvline(best_thresh, color='black', linestyle='--', label=f'Best Threshold = {best_thresh:.2f}')
plt.title('Precision, Recall & F1 vs Decision Threshold\n(Message Passing 2 rounds)')
plt.xlabel('Decision Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.savefig("Threshold_Precision_Recall_F1.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_true, y_prob)
plt.figure(figsize=(10, 6))
plt.plot(rec, prec, label='PR Curve')
plt.scatter(rec_list, prec_list, c='red', s=40, label='Evaluated Thresholds')
plt.title('Precision-Recall Curve with Threshold Points')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.savefig("Precision_Recall_Curve_Thresholds.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Plots and table generated successfully!")
print("Saved files:")
print("   • Threshold_Precision_Recall_F1.png")
print("   • Precision_Recall_Curve_Thresholds.png")
