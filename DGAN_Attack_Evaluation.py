# File: PDGAN_Attack_Evaluation_Fixed.py
import pandas as pd
import re
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
import random

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("=" * 90)
print("PDGAN ATTACK EVALUATION")
print("Using your strong baseline + matching MLM Generator")
print("=" * 90)


# ====================== STRONG BASELINE MODEL (Correct Adapter) ======================
class DistilBertEmail(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.adapter = nn.Sequential(
            nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.5), nn.Linear(192, 128)
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
            nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.5), nn.Linear(192, 128)
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


# Load strong baseline
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


# ====================== MATCHING PDGAN GENERATOR (MLM-based) ======================
class StrongPDGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlm = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

    def forward(self, input_ids, attention_mask=None):
        return self.mlm(input_ids=input_ids, attention_mask=attention_mask).logits


generator = StrongPDGAN_Generator().to(device)
generator.load_state_dict(torch.load("strong_pdgan_generator.pth", map_location=device))
generator.eval()
print("✅ PDGAN Generator (MLM) loaded successfully.")

# ====================== DATASET ======================
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset).rename(columns={"content": "email_text", "labels": "label"})
df = df[df['label'].isin([0, 1])].reset_index(drop=True)


def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else ""


df['url'] = df['email_text'].apply(extract_first_url)
df = df[df['url'] != ""].reset_index(drop=True)

_, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
test_df = test_df.reset_index(drop=True)

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")


# ====================== GENERATE ADVERSARIAL EMAILS ======================
def generate_pdgan_adv(text):
    text = str(text)
    if len(text) < 20:
        return text
    inputs = email_tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    input_ids = inputs.input_ids

    # Mask 25-30% tokens
    mask = torch.rand_like(input_ids, dtype=torch.float32, device=device) < 0.28
    mask = mask & (input_ids != email_tokenizer.pad_token_id)
    input_ids_masked = input_ids.clone()
    input_ids_masked[mask] = email_tokenizer.mask_token_id

    with torch.no_grad():
        logits = generator(input_ids_masked, inputs.attention_mask)
        fake_ids = logits.argmax(dim=-1)
        fake_ids = torch.where(mask, fake_ids, input_ids)

    adv_text = email_tokenizer.decode(fake_ids[0], skip_special_tokens=True)
    return adv_text


print("\nGenerating adversarial emails with PDGAN...")
test_adv = test_df.copy()
test_adv['adv_email'] = [generate_pdgan_adv(t) for t in tqdm(test_adv['email_text'])]


# ====================== EVALUATION ======================
def evaluate(df_eval, use_adv=False):
    y_true, y_prob = [], []
    col = 'adv_email' if use_adv else 'email_text'
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i + 8]
            emails = batch[col].tolist()
            urls = batch['url'].tolist()

            e_in = email_tokenizer(emails, padding=True, truncation=True, max_length=256, return_tensors="pt").to(
                device)
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
        "mcc": matthews_corrcoef(y_true, y_pred)
    }


clean = evaluate(test_df, use_adv=False)
adv = evaluate(test_adv, use_adv=True)

print("\n" + "=" * 80)
print("PDGAN ATTACK RESULTS")
print("=" * 80)
print(f"Clean Baseline   → F1: {clean['f1']:.4f} | Acc: {clean['accuracy']:.4f}")
print(f"After PDGAN Attack → F1: {adv['f1']:.4f} | Acc: {adv['accuracy']:.4f}")
print(f"F1 Drop          → {clean['f1'] - adv['f1']:.4f}")
print("\nEvaluation completed.")
