# File: Adversarial_LLM_Paraphrasing_Two_Prompts.py
import pandas as pd
import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import warnings
from tqdm import tqdm
import random

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("="*90)
print("LLM Paraphrasing Attack - Two Prompt Strategies")
print("1. Normal Prompt")
print("2. Strong / Aggressive Prompt")
print("="*90)

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

print("✅ Best model loaded.")

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

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

# ====================== TWO PROMPTS ======================
def paraphrase_with_prompt(text, model_name, prompt_style="normal"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    if prompt_style == "normal":
        prompt = f"Paraphrase the following email naturally: {text}"
    else:  # strong / aggressive
        prompt = f"Rewrite this email in a more convincing and phishing-like way while keeping the original meaning: {text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=512,
            num_beams=5,
            temperature=0.85,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.2
        )
    paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased

# ====================== EVALUATION ======================
def evaluate(use_adv=False):
    y_true, y_prob = [], []
    col = 'adv_email' if use_adv else 'email_text'
    with torch.no_grad():
        for i in range(0, len(test_df), 8):
            batch = test_df.iloc[i:i+8]
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

# ====================== RUN BOTH PROMPTS ======================
models = ["google/flan-t5-large", "facebook/bart-large"]

print("\nStarting LLM Paraphrasing Attacks with Two Prompt Styles...\n")
clean = evaluate(use_adv=False)
print(f"Clean Baseline → F1: {clean['f1']:.4f} | Acc: {clean['accuracy']:.4f}\n")

for model_name in models:
    for prompt_style in ["normal", "strong"]:
        name = f"{model_name.split('/')[-1]} ({prompt_style})"
        print(f"→ Attacking with {name}...")
        
        test_adv = test_df.copy()
        adv_emails = []
        
        for text in tqdm(test_adv['email_text'], desc=name):
            try:
                paraphrased = paraphrase_with_prompt(str(text), model_name, prompt_style)
                adv_emails.append(paraphrased)
            except:
                adv_emails.append(str(text))
        
        test_adv['adv_email'] = adv_emails
        adv_metrics = evaluate(test_adv, use_adv=True)
        
        print(f"   F1: {adv_metrics['f1']:.4f} | Drop: {clean['f1'] - adv_metrics['f1']:.4f}")

print("\nLLM Paraphrasing Experiment with Two Prompts Completed.")
