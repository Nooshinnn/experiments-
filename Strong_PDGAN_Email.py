# File: Strong_PDGAN_Email_Fixed.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import re
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from datasets import load_dataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("="*100)
print("STRONG PDGAN ADVERSARIAL GENERATOR")
print("Using your best strong baseline model")
print("="*100)

# ====================== STRONG BASELINE MODEL (Adapter version) ======================
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

# Load your strong model
checkpoint = torch.load("best_trained_model.pth", map_location=device)
email_model = DistilBertEmail().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing(rounds=2).to(device)

email_model.load_state_dict(checkpoint['email_model'])
url_model.load_state_dict(checkpoint['url_model'])
comm_model.load_state_dict(checkpoint['comm_model'])

# Freeze the discriminator (your strong model)
for p in list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()):
    p.requires_grad = False

email_model.eval()
url_model.eval()
comm_model.eval()

print("✅ Strong baseline model loaded successfully.")

# ====================== TOKENIZERS ======================
email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")
mlm_model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device)
mlm_model.eval()

# ====================== DATASET ======================
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset).rename(columns={"content": "email_text", "labels": "label"})
df = df[df['label'].isin([0, 1])].reset_index(drop=True)

def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else ""

df['url'] = df['email_text'].apply(extract_first_url)
df = df[df['url'] != ""].reset_index(drop=True)

# Use a small test set for faster PDGAN training
_, test_df = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=42)
test_df = test_df.reset_index(drop=True)

print(f"Using {len(test_df)} samples for PDGAN training.")

# ====================== STRONG PDGAN GENERATOR ======================
class StrongPDGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlm = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
    
    def forward(self, input_ids, attention_mask=None):
        return self.mlm(input_ids=input_ids, attention_mask=attention_mask).logits

generator = StrongPDGAN_Generator().to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.999))

# ====================== TRAINING LOOP ======================
print("\nStarting Strong PDGAN Training (Generator vs Frozen Strong Discriminator)...")

for epoch in range(30):   # Increase if you have time
    generator.train()
    total_g_loss = 0.0
    
    for i in tqdm(range(0, len(test_df), 8), desc=f"Epoch {epoch+1}/30"):
        batch = test_df.iloc[i:i+8]
        texts = batch['email_text'].tolist()
        
        # Tokenize real emails
        inputs = email_tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        
        # Generate adversarial version (mask important tokens and let MLM fill)
        mask_prob = 0.25
        mask = torch.rand(input_ids.shape) < mask_prob
        mask = mask & (input_ids != email_tokenizer.pad_token_id)
        input_ids_masked = input_ids.clone()
        input_ids_masked[mask] = email_tokenizer.mask_token_id
        
        # Generator prediction
        gen_logits = generator(input_ids_masked, inputs.attention_mask)
        
        # Sample tokens
        with torch.no_grad():
            fake_ids = gen_logits.argmax(dim=-1)
            fake_ids[~mask] = input_ids[~mask]   # Keep original non-masked tokens
        
        fake_texts = email_tokenizer.batch_decode(fake_ids, skip_special_tokens=True)
        
        # Get discriminator score on fake samples
        e_in = email_tokenizer(fake_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        d_output = comm_model(e_feat, u_feat)
        d_prob = torch.sigmoid(d_output)
        
        # Generator loss - fool the strong discriminator
        g_loss = -torch.mean(torch.log(d_prob + 1e-8))
        
        g_optimizer.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        g_optimizer.step()
        
        total_g_loss += g_loss.item()
    
    avg_loss = total_g_loss / (len(test_df) // 8 + 1)
    print(f"Epoch {epoch+1} | G Loss: {avg_loss:.4f}")

torch.save(generator.state_dict(), "strong_pdg an_generator.pth")
print("\n✅ Strong PDGAN Generator training completed and saved!")
