import pandas as pd
import re
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from datasets import load_dataset
import warnings
from tqdm import tqdm
import random

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 90)
print("ADVERSARIAL ATTACK: BAE (BERT-based Adversarial Examples)")
print("Using your strong baseline (best_trained_model.pth)")
print("=" * 90)


# ====================== STRONG BASELINE MODEL — matches PDGAN/URLGAN/PWWS exactly ======================
# FIXED: DistilBertWrapper (768->128, dropout 0.3) replaced with DistilBertEmail (768->192->128, dropout 0.5)
# FIXED: DomURLBERT and MessagePassing dropout 0.3 -> 0.5
class DistilBertEmail(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.adapter = nn.Sequential(
            nn.Linear(768, 192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(192, 128)
        )

    def forward(self, input_ids, attention_mask=None):
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
            nn.Dropout(0.5),
            nn.Linear(192, 128)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.adapter(pooled)


class MessagePassing(nn.Module):
    def __init__(self, rounds=2):
        super().__init__()
        self.rounds = rounds
        self.update_e = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5))
        self.update_u = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5))
        self.fc = nn.Linear(256, 1)

    def forward(self, e, u):
        for _ in range(self.rounds):
            e = self.update_e(torch.cat([e, u], dim=1))
            u = self.update_u(torch.cat([u, e], dim=1))
        return self.fc(torch.cat([e, u], dim=1)).squeeze(-1)


# FIXED: checkpoint name matches the strong baseline
checkpoint = torch.load("best_trained_model.pth", map_location=device)
email_model = DistilBertEmail().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing(rounds=2).to(device)

email_model.load_state_dict(checkpoint['email_model'])
url_model.load_state_dict(checkpoint['url_model'])
comm_model.load_state_dict(checkpoint['comm_model'])

email_model.eval()
url_model.eval()
comm_model.eval()
print("✅ Strong baseline loaded successfully.")

# ====================== TOKENIZERS ======================
email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

# ====================== DATASET — identical to URLGAN/PWWS pipeline ======================
# FIXED: rename first, then filter
# FIXED: drop empty URL rows (no synthetic injection)
# FIXED: test_size=0.15
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset).rename(columns={"content": "email_text", "labels": "label"})
df = df[df['label'].isin([0, 1])].reset_index(drop=True)


def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else ""


df['url'] = df['email_text'].apply(extract_first_url)
df = df[df['url'] != ""].reset_index(drop=True)

_, test_df = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=42)
test_df = test_df.reset_index(drop=True)
print(f"Test set size: {len(test_df)}")

# ====================== MLM FOR BAE ======================
mlm_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
mlm_model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device)
mlm_model.eval()


# ====================== BAE ATTACK ======================
def bae_attack(text, url, change_percent=0.30):
    """
    FIXED: accepts real URL so the discriminator's URL branch sees the correct
    signal during saliency scoring — dummy empty URL removed.
    FIXED: guard against [MASK] being truncated away (IndexError).
    """
    words = text.split()
    if len(words) < 5:
        return text

    num_changes = max(1, int(len(words) * change_percent))

    # Encode the real URL once — reused across all saliency queries
    u_in = url_tokenizer(
        [url], padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(device)

    # Original prediction
    with torch.no_grad():
        e_in = email_tokenizer(
            [text], padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(device)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        original_prob = torch.sigmoid(comm_model(e_feat, u_feat)).item()

    # Compute per-word saliency
    saliency_scores = []
    for i, word in enumerate(words):
        if not word.isalpha():
            saliency_scores.append(0.0)
            continue
        masked = words.copy()
        masked[i] = mlm_tokenizer.mask_token
        masked_text = " ".join(masked)

        with torch.no_grad():
            e_in = email_tokenizer(
                [masked_text], padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(device)
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            new_prob = torch.sigmoid(comm_model(e_feat, u_feat)).item()

        saliency_scores.append(abs(original_prob - new_prob))

    # Attack highest saliency words first
    indices = np.argsort(saliency_scores)[::-1][:num_changes]

    for idx in indices:
        if not words[idx].isalpha():
            continue
        masked = words.copy()
        masked[idx] = mlm_tokenizer.mask_token
        masked_text = " ".join(masked)

        inputs = mlm_tokenizer(
            masked_text, return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            logits = mlm_model(**inputs).logits

        # FIXED: guard against [MASK] being truncated away
        mask_positions = (inputs.input_ids == mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        if mask_positions.numel() == 0:
            continue
        mask_pos = mask_positions[0]
        top_tokens = logits[0, mask_pos].topk(40).indices

        for token_id in top_tokens:
            candidate = mlm_tokenizer.decode(token_id).strip()
            if candidate.isalpha() and candidate.lower() != words[idx].lower():
                words[idx] = candidate
                break

    return " ".join(words)


# ====================== EVALUATION ======================
def evaluate(df_eval, use_adv=False):
    y_true, y_prob = [], []
    col = 'adv_email' if use_adv else 'email_text'
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i + 8]
            e_in = email_tokenizer(
                batch[col].tolist(), padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            ).to(device)
            u_in = url_tokenizer(
                batch['url'].tolist(), padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(device)
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
        "mcc": matthews_corrcoef(y_true, y_pred)
    }


# ====================== RUN BAE WITH DIFFERENT PERCENTAGES ======================
percentages = [0.10, 0.20, 0.30, 0.40, 0.50]

print("\nRunning BAE attacks with different change percentages...\n")
clean = evaluate(test_df, use_adv=False)
print(f"Clean Baseline → F1: {clean['f1']:.4f} | Acc: {clean['accuracy']:.4f} | MCC: {clean['mcc']:.4f}")

for percent in percentages:
    print(f"\n→ Attacking with {percent * 100:.0f}% changes using BAE...")
    test_adv = test_df.copy()
    random.seed(42)  # reset seed per strength for reproducibility
    test_adv['adv_email'] = [
        bae_attack(str(row['email_text']), str(row['url']), change_percent=percent)
        for _, row in tqdm(test_adv.iterrows(), total=len(test_adv), desc=f"BAE {percent * 100:.0f}%")
    ]
    adv = evaluate(test_adv, use_adv=True)
    print(
        f"   F1: {adv['f1']:.4f} | F1 Drop: {clean['f1'] - adv['f1']:.4f} | "
        f"Acc: {adv['accuracy']:.4f} | Acc Drop: {clean['accuracy'] - adv['accuracy']:.4f} | "
        f"MCC: {adv['mcc']:.4f}"
    )

print("\nBAE attack experiment completed.")
