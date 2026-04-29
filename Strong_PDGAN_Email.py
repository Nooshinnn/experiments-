# File: Strong_PDGAN_Email.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import pandas as pd
import re
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== LOAD YOUR BEST MODEL (Discriminator) ======================
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

# Freeze Discriminator
for p in list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()):
    p.requires_grad = False

# ====================== STRONG PDGAN GENERATOR ======================
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
g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# ====================== DATA & TOKENIZER ======================
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset).rename(columns={"content": "email_text", "labels": "label"})
df = df[df['label'].isin([0,1])].reset_index(drop=True)

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Simple vocab for generator
all_text = " ".join(df['email_text'].astype(str))
vocab = sorted(set(all_text.lower()))
char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for i, c in enumerate(vocab)}

def text_to_seq(text, max_len=100):
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())[:max_len]
    return torch.tensor([char_to_idx.get(c, 0) for c in text], dtype=torch.long).to(device)

# ====================== TRAINING STRONG PDGAN ======================
print("Starting Strong PDGAN Training...")

for epoch in range(50):   # You can increase this
    generator.train()
    total_g_loss = 0.0
    
    for i in tqdm(range(0, len(df), 16), desc=f"Epoch {epoch+1}/50"):
        batch_texts = df.iloc[i:i+16]['email_text'].tolist()
        real_seq = torch.stack([text_to_seq(t) for t in batch_texts]).to(device)
        
        # Generate fake samples
        noise = torch.randint(0, len(vocab), (len(batch_texts), 80)).to(device)
        fake_logits = generator(noise)
        fake_seq = fake_logits.argmax(dim=-1)
        
        # Get discriminator score on fake samples
        fake_texts = ["".join([idx_to_char[idx.item()] for idx in sample]) for sample in fake_seq]
        e_in = email_tokenizer(fake_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer([""]*len(fake_texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        d_output = comm_model(e_feat, u_feat)
        d_prob = torch.sigmoid(d_output)
        
        # Generator loss (fool the discriminator)
        g_loss = -torch.mean(torch.log(d_prob + 1e-8))
        
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        total_g_loss += g_loss.item()
    
    print(f"Epoch {epoch+1} | G Loss: {total_g_loss/len(df):.4f}")

torch.save(generator.state_dict(), "strong_pdg an_generator.pth")
print("Strong PDGAN Generator saved!")
