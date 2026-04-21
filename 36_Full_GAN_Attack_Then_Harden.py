# File: 45_Full_Pipeline_Harshest_Attack_Harden_ReAttack.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from datasets import load_dataset
import warnings
from tqdm import tqdm
import numpy as np
import re
import random

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("="*160)
print("FULL PIPELINE: Clean → Harshest Attack → Hardening → Re-Attack")
print("Using best_trained_model.pth")
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

# Load saved model
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
print("Loaded trained model successfully.")

# ====================== LOAD DATA ======================
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset)
df = df.rename(columns={"content": "email_text", "labels": "label"})
df = df[df['label'].isin([0, 1])].reset_index(drop=True)

def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else ""

df['url'] = df['email_text'].apply(extract_first_url)
df = df[df['url'] != ""].reset_index(drop=True)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")
mlm_model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device)
mlm_model.eval()

# ====================== HARSHEST ATTACK (same as your successful script) ======================
def harshest_email_attack(text, num_changes=12, iterations=2):
    text = str(text)
    if len(text) < 30:
        return text
    for _ in range(iterations):
        enc = email_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        input_ids = enc["input_ids"][0]
        input_float = input_ids.float().unsqueeze(0).requires_grad_(True)
        with torch.enable_grad():
            outputs = email_model(input_float.long())
            loss = outputs.mean()
            loss.backward()
        importance = input_float.grad[0].abs().cpu().numpy() if input_float.grad is not None else np.ones(len(input_ids))
        tokens = email_tokenizer.convert_ids_to_tokens(input_ids.tolist())
        candidates = [(i, score) for i, score in enumerate(importance)
                      if not tokens[i].startswith("##") and tokens[i] not in email_tokenizer.all_special_tokens]
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = candidates[:num_changes]
        attacked_ids = input_ids.clone()
        for pos, _ in selected:
            masked = attacked_ids.clone()
            masked[pos] = email_tokenizer.mask_token_id
            with torch.no_grad():
                logits = mlm_model(input_ids=masked.unsqueeze(0)).logits[0, pos]
            top_tokens = torch.topk(logits, 15).indices
            for cand in top_tokens:
                if cand != attacked_ids[pos]:
                    attacked_ids[pos] = cand
                    break
        text = email_tokenizer.decode(attacked_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return text

def harshest_url_attack(url):
    url = str(url)
    if not url: return url
    homoglyphs = {'a':'а','e':'е','o':'о','c':'с','p':'р','i':'і','l':'І','s':'ѕ','b':'Ь','h':'н','k':'к'}
    for old, new in homoglyphs.items():
        url = url.replace(old, new)
    if random.random() < 0.9:
        url = url.replace("://", "://secure.login.")
    if random.random() < 0.7:
        url = url.replace(".", "%2E")
    if random.random() < 0.6:
        url = url + "/account-verification"
    if random.random() < 0.5:
        url = url.replace("www.", "ww w.")
    return url

# ====================== EVALUATION ======================
def evaluate(df_eval, use_adv=False):
    y_true, y_prob = [], []
    col_email = 'adv_email' if use_adv else 'email_text'
    col_url = 'adv_url' if use_adv else 'url'
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i+8]
            emails = [str(t) for t in batch[col_email].tolist()]
            urls = [str(u) for u in batch[col_url].tolist()]

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

# ====================== STAGE 1: Clean Baseline ======================
print("\n=== STAGE 1: Clean Baseline ===")
clean_baseline = evaluate(test_df, use_adv=False)
for k, v in clean_baseline.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

# ====================== STAGE 2: Generate Adversarial Samples (on BOTH train & test) ======================
print("\n=== STAGE 2: Generating Harshest Adversarial Samples ===")
test_df = test_df.copy()
test_df['adv_email'] = test_df['email_text'].apply(lambda x: harshest_email_attack(str(x), num_changes=12, iterations=2))
test_df['adv_url'] = test_df['url'].apply(harshest_url_attack)

train_df = train_df.copy()
train_df['adv_email'] = train_df['email_text'].apply(lambda x: harshest_email_attack(str(x), num_changes=12, iterations=2))
train_df['adv_url'] = train_df['url'].apply(harshest_url_attack)

adv_before = evaluate(test_df, use_adv=True)
print("Adversarial Performance BEFORE Hardening:")
for k, v in adv_before.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

print(f"\nDrop in F1 before hardening: {clean_baseline['f1'] - adv_before['f1']:.4f}")

# ====================== STAGE 3: HARDENING ======================
print("\n=== STAGE 3: Hardening with Original + Harshest Adversarial Samples ===")

main_optimizer = optim.AdamW(
    list(email_model.adapter.parameters()) +
    list(url_model.adapter.parameters()) +
    list(comm_model.parameters()),
    lr=5e-6, weight_decay=0.12
)

pos_weight = torch.tensor((1 - train_df['label'].mean()) / train_df['label'].mean()).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

for epoch in range(6):   # increased epochs for better hardening
    email_model.train()
    url_model.train()
    comm_model.train()
    total_loss = 0.0

    for i in tqdm(range(0, len(train_df), 8), desc=f"Hardening Epoch {epoch+1}"):
        batch = train_df.iloc[i:i+8]
        labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

        # Clean
        emails = [str(t) for t in batch['email_text'].tolist()]
        urls = [str(u) for u in batch['url'].tolist()]
        e_in = email_tokenizer(emails, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(urls, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        logits_clean = comm_model(e_feat, u_feat)
        loss_clean = criterion(logits_clean, labels)

        # Adversarial (harshest)
        adv_emails = batch['adv_email'].tolist()
        adv_urls = batch['adv_url'].tolist()
        e_in_adv = email_tokenizer(adv_emails, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in_adv = url_tokenizer(adv_urls, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        e_feat_adv = email_model(e_in_adv.input_ids, e_in_adv.attention_mask)
        u_feat_adv = url_model(u_in_adv.input_ids, u_in_adv.attention_mask)
        logits_adv = comm_model(e_feat_adv, u_feat_adv)
        loss_adv = criterion(logits_adv, labels)

        loss = loss_clean + 0.8 * loss_adv   # stronger adversarial weight

        main_optimizer.zero_grad()
        loss.backward()
        main_optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss / (len(train_df)//8 + 1):.4f}")

# Save hardened model
torch.save({
    'email_model': email_model.state_dict(),
    'url_model': url_model.state_dict(),
    'comm_model': comm_model.state_dict(),
}, "hardened_model.pth")
print("\nHardened model saved as 'hardened_model.pth'")

# ====================== STAGE 4: Re-Attack on Hardened Model ======================
print("\n=== STAGE 4: Re-Attack on Hardened Model ===")
adv_after = evaluate(test_df, use_adv=True)

print("\nPerformance AFTER Hardening (Adversarial):")
for k, v in adv_after.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

print(f"\nImprovement in F1 under attack: {adv_after['f1'] - adv_before['f1']:.4f}")

# Save results
results = pd.DataFrame([clean_baseline, adv_before, adv_after],
                       index=['Clean_Baseline', 'Adversarial_Before', 'Adversarial_After'])
results.to_csv("Full_Harshest_Hardening_Results.csv", index=True)
print("\nResults saved to Full_Harshest_Hardening_Results.csv")