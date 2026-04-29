# File: PDGAN_Attack_Evaluation.py
import pandas as pd
import re
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import warnings
from tqdm import tqdm
import random

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("PDGAN ATTACK EVALUATION")
print("Using trained generator to attack your best model")
print("="*80)

# ====================== LOAD YOUR BEST MODEL ======================
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

checkpoint = torch.load("my_best_model.pth", map_location=device)
email_model = DistilBertWrapper().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing(rounds=2).to(device)

email_model.load_state_dict(checkpoint['email_model'])
url_model.load_state_dict(checkpoint['url_model'])
comm_model.load_state_dict(checkpoint['comm_model'])

email_model.eval()
url_model.eval()
comm_model.eval()

# ====================== LOAD TRAINED PDGAN GENERATOR ======================
class StrongPDGAN_Generator(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.fc(out)

generator = StrongPDGAN_Generator().to(device)
generator.load_state_dict(torch.load("strong_pdg an_generator.pth", map_location=device))
generator.eval()

print("✅ PDGAN Generator loaded.")

# ====================== DATASET ======================
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset)
df = df[df['labels'].isin([0, 1])].reset_index(drop=True)
df = df.rename(columns={"labels": "label", "content": "email_text"})

_, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
test_df = test_df.reset_index(drop=True)

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

# ====================== GENERATE ADVERSARIAL SAMPLES ======================
def generate_pdgan_email(text, max_len=80):
    seq = torch.randint(0, 5000, (1, max_len)).to(device)
    with torch.no_grad():
        fake_logits = generator(seq)
        fake_idx = fake_logits.argmax(dim=-1)[0]
    fake_text = "".join([chr(97 + (i % 26)) for i in fake_idx])  # simple mapping
    return fake_text[:len(text)]  # keep similar length

print("\nGenerating adversarial emails using PDGAN...")
test_adv = test_df.copy()
test_adv['adv_email'] = [generate_pdgan_email(str(t)) for t in tqdm(test_adv['email_text'])]

# ====================== EVALUATION ======================
def evaluate(df_eval, use_adv=False):
    y_true, y_prob = [], []
    col = 'adv_email' if use_adv else 'email_text'
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i+8]
            e_in = email_tokenizer(batch[col].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
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

print("\n" + "="*70)
print("PDGAN ATTACK RESULTS")
print("="*70)
print(f"Clean Baseline     → F1: {clean['f1']:.4f} | Acc: {clean['accuracy']:.4f}")
print(f"After PDGAN Attack → F1: {adv['f1']:.4f} | Acc: {adv['accuracy']:.4f}")
print(f"F1 Drop            → {clean['f1'] - adv['f1']:.4f}")

print("\nAttack completed.")
