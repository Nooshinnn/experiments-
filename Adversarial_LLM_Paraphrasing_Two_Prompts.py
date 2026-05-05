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
print("=" * 100)
print("ADVERSARIAL HARDENING PIPELINE - STRONG PROMPT (English Only)")
print("Model: best_pretrained_model")
print("=" * 100)

# ====================== SETUP ======================
print("\nLoading best pretrained model...")

class DistilBertEmail(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.adapter = nn.Sequential(
            nn.Linear(768, 192), nn.ReLU(), nn.Dropout(dropout), nn.Linear(192, 128)
        )
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.adapter(outputs.last_hidden_state[:, 0, :])

class DomURLBERT(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.model = AutoModel.from_pretrained("amahdaouy/DomURLs_BERT")
        self.adapter = nn.Sequential(
            nn.Linear(768, 192), nn.ReLU(), nn.Dropout(dropout), nn.Linear(192, 128)
        )
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

# Load your saved model
model_dir = "best_pretrained_model"
email_model = DistilBertEmail().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing(rounds=2).to(device)

email_model.load_state_dict(torch.load(f"{model_dir}/email_model.pth", map_location=device))
url_model.load_state_dict(torch.load(f"{model_dir}/url_model.pth", map_location=device))
comm_model.load_state_dict(torch.load(f"{model_dir}/comm_model.pth", map_location=device))

email_model.eval()
url_model.eval()
comm_model.eval()
print("Best pretrained model loaded successfully.")

# ====================== LOAD DATASET ======================
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

print(f"Total usable samples: {len(df)}")
print(f"Test samples: {len(test_df)}")

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

# ====================== STRONG PROMPT PARAPHRASER ======================
print("\nLoading FLAN-T5-Large for strong prompt attack...")
paraphraser = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
para_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

def strong_paraphrase(text):
    prompt = (
        "Rewrite the following email to sound highly professional, natural, trustworthy, "
        "and convincing while keeping the exact same meaning and intent:\n\n"
        f"{text}\n\nRewritten email:"
    )
    inputs = para_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = paraphraser.generate(
            **inputs, max_length=512, num_beams=6, temperature=0.7, 
            top_p=0.92, do_sample=True, early_stopping=True
        )
    paraphrased = para_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased if paraphrased.strip() else text

# ====================== EVALUATION FUNCTION ======================
def evaluate(df_eval, use_adv=False, batch_size=8):
    y_true = []
    y_prob = []
    col = "adv_email" if use_adv else "email_text"
    with torch.no_grad():
        for i in range(0, len(df_eval), batch_size):
            batch = df_eval.iloc[i:i + batch_size]
            emails = [str(t) for t in batch[col].tolist()]
            urls = [str(u) for u in batch["url"].tolist()]
            
            e_in = email_tokenizer(emails, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(urls, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            prob = logits.sigmoid().detach().cpu().numpy()
            
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

# ====================== STAGE 1: CLEAN BASELINE ======================
print("\n=== STAGE 1: CLEAN BASELINE ===")
clean_metrics = evaluate(test_df, use_adv=False)
for k, v in clean_metrics.items():
    print(f"{k.replace('_', ' ').title()}: {v:.4f}")

# ====================== STAGE 2: STRONG PROMPT ATTACK ======================
print("\n=== STAGE 2: STRONG PROMPT ATTACK ===")
test_adv = test_df.copy()
adv_emails = [strong_paraphrase(text) for text in tqdm(test_adv["email_text"], desc="Strong Prompt Attack")]
test_adv["adv_email"] = adv_emails
test_adv.to_csv("strong_adversarial_test_set.csv", index=False)
print("Strong adversarial samples saved to: strong_adversarial_test_set.csv")

adv_metrics_before = evaluate(test_adv, use_adv=True)
print(f"Adversarial Accuracy: {adv_metrics_before['accuracy']:.4f}")
print(f"Adversarial F1: {adv_metrics_before['f1']:.4f}")
print(f"F1 Drop: {clean_metrics['f1'] - adv_metrics_before['f1']:.4f}")

# ====================== STAGE 3: ADVERSARIAL HARDENING ======================
print("\n=== STAGE 3: ADVERSARIAL HARDENING (Retraining) ===")
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
    total_loss = 0.0
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
    print(f"Epoch {epoch+1} - Avg Loss: {total_loss / (len(hardening_df)//6 + 1):.4f}")

# Save hardened model
save_dir = "hardened_strong_prompt_model"
os.makedirs(save_dir, exist_ok=True)
torch.save(email_model.state_dict(), f"{save_dir}/email_model.pth")
torch.save(url_model.state_dict(), f"{save_dir}/url_model.pth")
torch.save(comm_model.state_dict(), f"{save_dir}/comm_model.pth")
print(f"Hardened model saved to: {save_dir}/")

# ====================== STAGE 4: RE-EVALUATE AFTER HARDENING ======================
print("\n=== STAGE 4: PERFORMANCE AFTER HARDENING ===")
adv_metrics_after = evaluate(test_adv, use_adv=True)

print(f"Clean Accuracy          : {clean_metrics['accuracy']:.4f} | F1: {clean_metrics['f1']:.4f}")
print(f"Before Hardening        : {adv_metrics_before['accuracy']:.4f} | F1: {adv_metrics_before['f1']:.4f}")
print(f"After Hardening         : {adv_metrics_after['accuracy']:.4f} | F1: {adv_metrics_after['f1']:.4f}")
print(f"Recovery in F1          : +{adv_metrics_after['f1'] - adv_metrics_before['f1']:.4f}")

# ====================== SAVE FINAL RESULTS ======================
results_summary = pd.DataFrame({
    "Stage": ["Clean", "Strong Attack (Before)", "After Hardening"],
    "Accuracy": [clean_metrics['accuracy'], adv_metrics_before['accuracy'], adv_metrics_after['accuracy']],
    "F1": [clean_metrics['f1'], adv_metrics_before['f1'], adv_metrics_after['f1']],
    "MCC": [clean_metrics['mcc'], adv_metrics_before['mcc'], adv_metrics_after['mcc']]
})
results_summary.to_csv("strong_prompt_hardening_full_results.csv", index=False)

print("\n=== EXPERIMENT COMPLETED SUCCESSFULLY ===")
print("Files saved:")
print("- strong_adversarial_test_set.csv")
print("- hardened_strong_prompt_model/")
print("- strong_prompt_hardening_full_results.csv")
