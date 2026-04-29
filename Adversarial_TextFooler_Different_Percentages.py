# File: Adversarial_TextFooler_Different_Percentages.py
import pandas as pd
import re
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from datasets import load_dataset
import warnings
from tqdm import tqdm
import random

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*90)
print("ADVERSARIAL ATTACK: TextFooler with Different Change Percentages")
print("Using your saved best model: my_best_model.pth")
print("="*90)

# ====================== LOAD SAVED MODEL ======================
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

# Load your best model
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

print("✅ Best model loaded successfully.")

# ====================== DATASET ======================
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset)
df = df[df['labels'].isin([0, 1])].reset_index(drop=True)
df = df.rename(columns={"labels": "label", "content": "email_text"})

def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else ""

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

# Use test set
from sklearn.model_selection import train_test_split
_, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
test_df = test_df.reset_index(drop=True)

# ====================== TEXTFOOLER ATTACK ======================
mlm_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
mlm_model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device)

def textfooler_attack(text, change_percent=0.30):
    words = text.split()
    if len(words) < 5:
        return text
    num_changes = max(1, int(len(words) * change_percent))
    
    changed = 0
    for i in random.sample(range(len(words)), len(words)):
        if changed >= num_changes:
            break
        if not words[i].isalpha():
            continue
            
        masked = words.copy()
        masked[i] = mlm_tokenizer.mask_token
        masked_text = " ".join(masked)
        
        inputs = mlm_tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = mlm_model(**inputs).logits
        mask_idx = (inputs.input_ids == mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[1][0]
        top_tokens = logits[0, mask_idx].topk(40).indices
        
        for token_id in top_tokens:
            candidate = mlm_tokenizer.decode(token_id)
            if candidate.isalpha() and candidate.lower() != words[i].lower():
                words[i] = candidate
                changed += 1
                break
    return " ".join(words)

def evaluate(model_email, model_url, model_comm, df_eval, use_adv=False):
    y_true, y_prob = [], []
    col_email = 'adv_email' if use_adv else 'email_text'
    col_url = 'adv_url' if use_adv else 'url'
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i+8]
            e_in = email_tokenizer(batch[col_email].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(batch[col_url].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            e_feat = model_email(e_in.input_ids, e_in.attention_mask)
            u_feat = model_url(u_in.input_ids, u_in.attention_mask)
            logits = model_comm(e_feat, u_feat)
            prob = logits.sigmoid().cpu().numpy()
            y_true.extend(batch['label'].values)
            y_prob.extend(prob)
    y_pred = (np.array(y_prob) > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

# ====================== RUN ATTACKS WITH DIFFERENT PERCENTAGES ======================
percentages = [0.10, 0.20, 0.30, 0.40, 0.50]

print("\nRunning TextFooler attacks with different change percentages...\n")
clean_metrics = evaluate(email_model, url_model, comm_model, test_df, use_adv=False)
print(f"Clean Baseline - F1: {clean_metrics['f1']:.4f} | Accuracy: {clean_metrics['accuracy']:.4f}")

results = []

for percent in percentages:
    print(f"\nAttacking with {percent*100:.0f}% word changes...")
    test_adv = test_df.copy()
    test_adv['adv_email'] = [textfooler_attack(str(t), change_percent=percent) for t in tqdm(test_adv['email_text'], desc=f"{percent*100:.0f}%")]
    test_adv['adv_url'] = test_adv['url']  # Keep URL same for this experiment (or add URL attack later)

    adv_metrics = evaluate(email_model, url_model, comm_model, test_adv, use_adv=True)
    
    print(f"  F1 Drop: {clean_metrics['f1'] - adv_metrics['f1']:.4f} | Acc Drop: {clean_metrics['accuracy'] - adv_metrics['accuracy']:.4f}")
    results.append((percent, adv_metrics['f1'], adv_metrics['accuracy']))

print("\n" + "="*60)
print("TEXTFOOLER RESULTS SUMMARY")
print("="*60)
print("Change % | F1 Score | F1 Drop")
for p, f1, _ in results:
    print(f"{p*100:6.0f}%   | {f1:.4f}   | {clean_metrics['f1'] - f1:.4f}")

print(f"\nBest performing attack (least damage): {min(results, key=lambda x: clean_metrics['f1'] - x[1])[0]*100:.0f}% change")
