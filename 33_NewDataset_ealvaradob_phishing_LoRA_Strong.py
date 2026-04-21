# File: 33_zefang-liu_phishing_email_LoRA_Strong_Fixed.py
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
import re

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("="*130)
print("GENERALIZATION TEST ON zefang-liu/phishing-email-dataset")
print("Model: DistilBERT (Email) + DomURLBERT (URL) + Message Passing")
print("LoRA-style + Strong Freezing + High Regularization")
print("="*130)

# ====================== MODEL ======================
class DistilBertEmail(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.adapter = nn.Sequential(
            nn.Linear(768, 192),
            nn.ReLU(),
            nn.Dropout(0.55),
            nn.Linear(192, 128)
        )
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.adapter(pooled)

class DomURLBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("amahdaouy/DomURLs_BERT")
        self.adapter = nn.Sequential(
            nn.Linear(768, 192),
            nn.ReLU(),
            nn.Dropout(0.55),
            nn.Linear(192, 128)
        )
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.adapter(pooled)

class MessagePassing(nn.Module):
    def __init__(self, dim=128, rounds=2):
        super().__init__()
        self.rounds = rounds
        self.update_e = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Dropout(0.55))
        self.update_u = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Dropout(0.55))
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

# Fix column names
df = df.rename(columns={
    "Email Text": "email_text",
    "Email Type": "label"
})

# Convert label to numeric
label_map = {"Safe Email": 0, "Phishing Email": 1}
df['label'] = df['label'].map(label_map)

# Clean email_text (remove NaN and non-string values)
df = df.dropna(subset=['email_text'])
df['email_text'] = df['email_text'].astype(str)

# Extract URL
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

# ====================== TOKENIZERS ======================
email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

# ====================== MODELS ======================
email_model = DistilBertEmail().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing().to(device)

trainable_params = list(email_model.adapter.parameters()) + \
                   list(url_model.adapter.parameters()) + \
                   list(comm_model.parameters())

optimizer = optim.AdamW(trainable_params, lr=5e-6, weight_decay=0.15)

pos_weight = torch.tensor((1 - train_df['label'].mean()) / train_df['label'].mean()).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ====================== TRAINING ======================
batch_size = 8
max_epochs = 15
patience = 5
best_val_f1 = 0.0
patience_counter = 0

print("\nStarting ultra-conservative LoRA-style fine-tuning...\n")

for epoch in range(max_epochs):
    email_model.train()
    url_model.train()
    comm_model.train()

    if epoch < 6:
        for p in email_model.model.parameters(): p.requires_grad = False
        for p in url_model.model.parameters(): p.requires_grad = False

    train_loss = 0.0
    progress = tqdm(range(0, len(train_df), batch_size), desc=f"Epoch {epoch+1}/{max_epochs}")

    for i in progress:
        batch = train_df.iloc[i:i+batch_size]
        labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

        # Clean batch (extra safety)
        email_texts = [str(text) for text in batch['email_text'].tolist()]
        urls = [str(url) for url in batch['url'].tolist()]

        e_in = email_tokenizer(email_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(urls, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)

        logits = comm_model(e_feat, u_feat)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress.set_postfix({'Loss': f'{train_loss / (i//batch_size + 1):.4f}'})

    # Validation
    email_model.eval()
    url_model.eval()
    comm_model.eval()
    y_true_val, y_prob_val = [], []

    with torch.no_grad():
        for i in range(0, len(val_df), batch_size):
            batch = val_df.iloc[i:i+batch_size]
            email_texts = [str(text) for text in batch['email_text'].tolist()]
            urls = [str(url) for url in batch['url'].tolist()]

            e_in = email_tokenizer(email_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(urls, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

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
            print("Early stopping triggered!")
            break

# ====================== FINAL TEST EVALUATION ======================
print("\n=== FINAL TEST SET EVALUATION (True Held-out Generalization) ===")
email_model.eval()
url_model.eval()
comm_model.eval()

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

final_metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    "f1": f1_score(y_true, y_pred, zero_division=0),
    "mcc": matthews_corrcoef(y_true, y_pred),
    "roc_auc": roc_auc_score(y_true, y_prob),
}

print("\nTest Set Results:")
for k, v in final_metrics.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Legitimate (0)', 'Phishing (1)'], digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

pd.DataFrame([final_metrics]).to_csv("zefang_liu_phishing_LoRA_Strong_Results.csv", index=False)
print("\nResults saved to zefang_liu_phishing_LoRA_Strong_Results.csv")
