
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
print("ADVERSARIAL HARDENING PIPELINE - STRONGEST PROMPT")
print("Using checkpoint: best_trained_model.pth")
print(f"Device: {device}")
print("=" * 100)


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


print("\nLoading best_trained_model.pth ...")
checkpoint = torch.load("best_trained_model.pth", map_location=device)

email_model = DistilBertEmail().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing(rounds=2).to(device)

email_model.load_state_dict(checkpoint["email_model"])
url_model.load_state_dict(checkpoint["url_model"])
comm_model.load_state_dict(checkpoint["comm_model"])

email_model.eval()
url_model.eval()
comm_model.eval()
print("✅ Model loaded successfully!")


dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset)
df = df.rename(columns={"content": "email_text", "labels": "label"})
df = df[df["label"].isin([0, 1])].reset_index(drop=True)


def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else ""


df["url"] = df["email_text"].apply(extract_first_url)
df = df[df["url"] != ""].reset_index(drop=True)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
test_df = test_df.reset_index(drop=True)

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

print("\nLoading FLAN-T5-Large...")
paraphraser = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
para_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")


def strongest_paraphrase(text):
    prompt = (
        "You are a highly skilled professional writer. Rewrite the following email to be "
        "extremely convincing, natural, formal, and trustworthy. Make it sound like an "
        "important legitimate business or personal message. Use polite, urgent, and "
        "professional language. Keep the exact same meaning and intent but make it "
        "very difficult to detect as suspicious or phishing:\n\n"
        f"{text}\n\n"
        "Professional Rewritten Email:"
    )
    inputs = para_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = paraphraser.generate(
            **inputs,
            max_length=512,
            num_beams=10,
            temperature=0.95,
            top_p=0.98,
            do_sample=True,
            early_stopping=True,
            repetition_penalty=1.25
        )
    return para_tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate(df_eval, use_adv=False, batch_size=8):
    y_true = []
    y_prob = []
    col = "adv_email" if use_adv else "email_text"
    with torch.no_grad():
        for i in range(0, len(df_eval), batch_size):
            batch = df_eval.iloc[i:i + batch_size]
            emails = [str(t) for t in batch[col].tolist()]
            urls = [str(u) for u in batch["url"].tolist()]

            e_in = email_tokenizer(emails, padding=True, truncation=True, max_length=256, return_tensors="pt").to(
                device)
            u_in = url_tokenizer(urls, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            prob = torch.sigmoid(logits).detach().cpu().numpy()

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




clean_metrics = evaluate(test_df, use_adv=False)
for k, v in clean_metrics.items():
    print(f"{k.replace('_', ' ').title()}: {v:.4f}")


test_adv = test_df.copy()
adv_emails = [strongest_paraphrase(text) for text in tqdm(test_adv["email_text"], desc="Strongest Attack")]
test_adv["adv_email"] = adv_emails
test_adv.to_csv("strongest_adversarial_test_set.csv", index=False)

adv_before = evaluate(test_adv, use_adv=True)
print(f"\nAfter Strongest Attack → Accuracy: {adv_before['accuracy']:.4f} | F1: {adv_before['f1']:.4f}")
print(f"F1 Drop: {clean_metrics['f1'] - adv_before['f1']:.4f}")


print("\n=== STAGE 3: ADVERSARIAL HARDENING ===")

hardening_df = pd.concat([
    train_df[['email_text', 'label', 'url']].assign(source='original'),
    test_adv[['adv_email', 'label', 'url']].rename(columns={'adv_email': 'email_text'}).assign(source='adversarial')
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

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
    for i in tqdm(range(0, len(hardening_df), 6), desc=f"Hardening Epoch {epoch + 1}"):
        batch = hardening_df.iloc[i:i + 6]
        labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

        e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256,
                               return_tensors="pt").to(device)
        u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128,
                             return_tensors="pt").to(device)

        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        logits = comm_model(e_feat, u_feat)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()), 1.0)
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1} - Avg Loss: {total_loss / (len(hardening_df) // 6 + 1):.4f}")


save_dir = "hardened_strongest_prompt_model"
os.makedirs(save_dir, exist_ok=True)
torch.save(email_model.state_dict(), f"{save_dir}/email_model.pth")
torch.save(url_model.state_dict(), f"{save_dir}/url_model.pth")
torch.save(comm_model.state_dict(), f"{save_dir}/comm_model.pth")
print(f"Hardened model saved to: {save_dir}/")


adv_after = evaluate(test_adv, use_adv=True)

print(f"Clean          : Acc {clean_metrics['accuracy']:.4f} | F1 {clean_metrics['f1']:.4f}")
print(f"Before Hardening: Acc {adv_before['accuracy']:.4f} | F1 {adv_before['f1']:.4f}")
print(f"After Hardening : Acc {adv_after['accuracy']:.4f} | F1 {adv_after['f1']:.4f}")
print(f"Recovery        : +{adv_after['f1'] - adv_before['f1']:.4f} F1")

summary = pd.DataFrame({
    "Stage": ["Clean", "Strongest Attack", "After Hardening"],
    "Accuracy": [clean_metrics['accuracy'], adv_before['accuracy'], adv_after['accuracy']],
    "F1": [clean_metrics['f1'], adv_before['f1'], adv_after['f1']]
})
summary.to_csv("strongest_prompt_hardening_summary.csv", index=False)

