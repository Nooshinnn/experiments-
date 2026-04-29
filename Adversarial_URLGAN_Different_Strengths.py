# File: Adversarial_URLGAN_Different_Strengths.py
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
import string

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*90)
print("ADVERSARIAL ATTACK: URLGAN-style (Character + Structural URL Attack)")
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

_, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
test_df = test_df.reset_index(drop=True)

# ====================== URLGAN-STYLE ATTACK ======================
def urlgan_attack(url, strength=0.3):
    if not url or len(url) < 10:
        return url
    
    # Strength controls how aggressive the attack is
    num_changes = max(2, int(len(url) * strength))
    
    url_list = list(url)
    
    for _ in range(num_changes):
        op = random.choice(['homoglyph', 'subdomain', 'insert', 'replace'])
        
        if op == 'homoglyph':
            homoglyph_map = {'a':'а', 'e':'е', 'o':'о', 'c':'с', 'l':'1', '1':'l', '.':'.'}
            for i in range(len(url_list)):
                if url_list[i] in homoglyph_map and random.random() < 0.6:
                    url_list[i] = homoglyph_map[url_list[i]]
                    break
                    
        elif op == 'subdomain' and '://' in ''.join(url_list):
            pos = ''.join(url_list).find('://') + 3
            if pos < len(url_list):
                url_list.insert(pos, random.choice(['secure-', 'login.', 'account.']))
                
        elif op == 'insert':
            pos = random.randint(0, len(url_list)-1)
            url_list.insert(pos, random.choice(string.ascii_lowercase + '0123456789'))
            
        elif op == 'replace':
            pos = random.randint(0, len(url_list)-1)
            if url_list[pos].isalnum():
                url_list[pos] = random.choice(string.ascii_lowercase)
    
    return ''.join(url_list)

# ====================== EVALUATION ======================
def evaluate(use_adv=False):
    y_true, y_prob = [], []
    col_url = 'adv_url' if use_adv else 'url'
    with torch.no_grad():
        for i in range(0, len(test_df), 8):
            batch = test_df.iloc[i:i+8]
            e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(batch[col_url].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
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

# ====================== RUN URLGAN WITH DIFFERENT STRENGTHS ======================
strengths = [0.1, 0.2, 0.3, 0.4, 0.5]

print("\nRunning URLGAN-style attacks...\n")
clean = evaluate(use_adv=False)
print(f"Clean Baseline → F1: {clean['f1']:.4f} | Acc: {clean['accuracy']:.4f}")

for strength in strengths:
    print(f"\n→ Attacking URLs with strength {strength*100:.0f}% (URLGAN)...")
    test_adv = test_df.copy()
    test_adv['adv_url'] = [urlgan_attack(str(u), strength) for u in tqdm(test_adv['url'], desc=f"URLGAN {strength*100:.0f}%")]
    
    adv = evaluate(use_adv=True)
    print(f"   F1: {adv['f1']:.4f} | F1 Drop: {clean['f1'] - adv['f1']:.4f} | Acc Drop: {clean['accuracy'] - adv['accuracy']:.4f}")

print("\nURLGAN attack experiment completed.")
