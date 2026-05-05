# File: 30_Adversarial_Hardening_Strong_Prompt_Full.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import re
import random
import numpy as np
import gc
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("=" * 100)
print("ADVERSARIAL HARDENING PIPELINE - STRONG PROMPT")
print("Step 1: Attack → Step 2: Harden → Step 3: Re-evaluate")
print("=" * 100)

# ====================== LOAD BEST MODEL ======================
print("\nLoading best trained model...")
checkpoint_dir = "best_optimized_phishing_model"

class DistilBertEmail(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.adapter = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(dropout), nn.Linear(192, 128))
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.adapter(outputs.last_hidden_state[:, 0, :])

class DomURLBERT(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.model = AutoModel.from_pretrained("amahdaouy/DomURLs_BERT")
        self.adapter = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(dropout), nn.Linear(192, 128))
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.adapter(outputs.last_hidden_state[:, 0, :])

class MessagePassing(nn.Module):
    def __init__(self, rounds=2):
        super().__init__()
        self.rounds = rounds
        self.update_e = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4))
        self.update_u = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4))
        self.fc = nn.Linear(256, 1)
    def forward(self, e, u):
        for _ in range(self.rounds):
            e = self.update_e(torch.cat([e, u], dim=1))
            u = self.update_u(torch.cat([u, e], dim=1))
        return self.fc(torch.cat([e, u], dim=1)).squeeze(-1)

email_model = DistilBertEmail().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing(rounds=2).to(device)

email_model.load_state_dict(torch.load(f"{checkpoint_dir}/email_model.pth", map_location=device))
url_model.load_state_dict(torch.load(f"{checkpoint_dir}/url_model.pth", map_location=device))
comm_model.load_state_dict(torch.load(f"{checkpoint_dir}/comm_model.pth", map_location=device))

email_model.eval()
url_model.eval()
comm_model.eval()
print("Best classifier loaded.")

# ====================== LOAD DATA ======================
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset)
df = df.rename(columns={"content": "email_text", "labels": "label"})
df = df[df["label"].isin([0, 1])].reset_index(drop=True)

def extract_first_url(text):
    urls = re.findall(r"https?://\S+", str(text))
    return urls[0] if urls else ""

df["url"] = df["email_text"].apply(extract_first_url)
df = df[df["url"] != ""].reset_index(drop=True)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
test_df = test_df.reset_index(drop=True)

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

# ====================== STRONG PROMPT ATTACK ======================
print("\nLoading FLAN-T5-Large for strong attack...")
paraphraser = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
para_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

def strong_paraphrase(text):
    prompt = (
        "Rewrite the following email to sound highly professional, natural, trustworthy, "
        "and convincing while keeping the exact same meaning:\n\n"
        f"{text}\n\nRewritten email:"
    )
    inputs = para_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = paraphraser.generate(**inputs, max_length=512, num_beams=6, temperature=0.7, top_p=0.92, do_sample=True)
    return para_tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerating strong adversarial samples...")
test_adv = test_df.copy()
adv_emails = [strong_paraphrase(text) for text in tqdm(test_adv["email_text"], desc="Attacking")]
test_adv["adv_email"] = adv_emails
test_adv.to_csv("strong_adversarial_test_set.csv", index=False)

# ====================== STAGE 1: ATTACK BEFORE HARDENING ======================
def evaluate(df_eval, use_adv=False):
    email_model.eval()
    url_model.eval()
    comm_model.eval()
    y_true, y_prob = [], []
    col = "adv_email" if use_adv else "email_text"
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i+8]
            e_in = email_tokenizer([str(x) for x in batch[col]], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer([str(x) for x in batch["url"]], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            prob = torch.sigmoid(logits).cpu().numpy()
            y_true.extend(batch["label"].values)
            y_prob.extend(prob)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

print("\n=== STAGE 1: Performance BEFORE Hardening ===")
clean_metrics = evaluate(test_df, use_adv=False)
adv_metrics_before = evaluate(test_adv, use_adv=True)

print(f"Clean Accuracy : {clean_metrics['accuracy']:.4f} | F1: {clean_metrics['f1']:.4f}")
print(f"Adversarial Accuracy : {adv_metrics_before['accuracy']:.4f} | F1: {adv_metrics_before['f1']:.4f}")
print(f"→ F1 Drop: {clean_metrics['f1'] - adv_metrics_before['f1']:.4f}")

# ====================== STAGE 2: ADVERSARIAL HARDENING ======================
print("\n=== STAGE 2: Adversarial Hardening (Retraining) ===")
hardening_df = pd.concat([
    train_df.assign(email_text=train_df['email_text']),
    test_adv.rename(columns={'adv_email': 'email_text'})
]).sample(frac=1, random_state=42).reset_index(drop=True)

optimizer = optim.AdamW(
    list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()),
    lr=5e-5, weight_decay=0.01
)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(6):
    email_model.train()
    url_model.train()
    comm_model.train()
    total_loss = 0
    for i in tqdm(range(0, len(hardening_df), 6), desc=f"Hardening Epoch {epoch+1}"):
        batch = hardening_df.iloc[i:i+6]
        labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)
        
        e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        logits = comm_model(e_feat, u_feat)
        
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()), 1.0)
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss / (len(hardening_df)//6 + 1):.4f}")

# Save hardened model
save_dir = "hardened_strong_prompt_model"
os.makedirs(save_dir, exist_ok=True)
torch.save(email_model.state_dict(), f"{save_dir}/email_model.pth")
torch.save(url_model.state_dict(), f"{save_dir}/url_model.pth")
torch.save(comm_model.state_dict(), f"{save_dir}/comm_model.pth")
print(f"Hardened model saved to: {save_dir}/")

# ====================== STAGE 3: RE-EVALUATE AFTER HARDENING ======================
print("\n=== STAGE 3: Performance AFTER Hardening ===")
adv_metrics_after = evaluate(test_adv, use_adv=True)

print(f"Clean Accuracy          : {clean_metrics['accuracy']:.4f}")
print(f"Before Hardening (Adv)  : {adv_metrics_before['accuracy']:.4f} | F1: {adv_metrics_before['f1']:.4f}")
print(f"After Hardening (Adv)   : {adv_metrics_after['accuracy']:.4f} | F1: {adv_metrics_after['f1']:.4f}")
print(f"Recovery in F1          : {adv_metrics_after['f1'] - adv_metrics_before['f1']:.4f}")

# Save summary
summary = pd.DataFrame({
    "Stage": ["Clean", "Before Hardening", "After Hardening"],
    "Accuracy": [clean_metrics['accuracy'], adv_metrics_before['accuracy'], adv_metrics_after['accuracy']],
    "F1": [clean_metrics['f1'], adv_metrics_before['f1'], adv_metrics_after['f1']]
})
summary.to_csv("adversarial_hardening_strong_prompt_summary.csv", index=False)
print("\nAll results saved successfully!")
