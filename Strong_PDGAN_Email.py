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

print("=" * 100)
print("STRONG PDGAN - STABLE FINAL VERSION")
print("Using your best baseline model")
print("=" * 100)


# ====================== YOUR STRONG BASELINE ======================
class DistilBertEmail(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.adapter = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.5), nn.Linear(192, 128))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.adapter(pooled)

    def forward_soft(self, soft_embeds, attention_mask=None):
        """
        Accept soft (continuous) token embeddings instead of discrete input_ids.
        soft_embeds: [batch, seq_len, vocab_size] — Gumbel-Softmax probabilities
        We multiply by the embedding matrix to get [batch, seq_len, hidden_size],
        keeping gradients alive all the way back to the generator.
        """
        embedding_matrix = self.model.embeddings.word_embeddings.weight  # [vocab_size, 768]
        # [B, T, vocab] @ [vocab, 768] -> [B, T, 768]
        inputs_embeds = torch.matmul(soft_embeds, embedding_matrix)

        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.adapter(pooled)


class DomURLBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("amahdaouy/DomURLs_BERT")
        self.adapter = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.5), nn.Linear(192, 128))

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


checkpoint = torch.load("best_trained_model.pth", map_location=device)
email_model = DistilBertEmail().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing(rounds=2).to(device)

email_model.load_state_dict(checkpoint['email_model'])
url_model.load_state_dict(checkpoint['url_model'])
comm_model.load_state_dict(checkpoint['comm_model'])

# Freeze discriminator parameters — but keep requires_grad=True on
# the embedding matrix so soft embedding matmul can backprop through it.
for name, p in list(email_model.named_parameters()) + \
               list(url_model.named_parameters()) + \
               list(comm_model.named_parameters()):
    p.requires_grad = False

# UN-freeze just the word embedding weights of the email discriminator so that
# the matmul in forward_soft produces a grad_fn. The weights themselves won't
# move (optimizer only covers generator params) but autograd needs this path.
email_model.model.embeddings.word_embeddings.weight.requires_grad = True

email_model.eval()
url_model.eval()
comm_model.eval()
print("✅ Strong baseline loaded and frozen (embedding path kept for soft backprop).")

# ====================== TOKENIZERS ======================
email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

# ====================== DATA ======================
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


# ====================== GENERATOR ======================
class PDGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlm = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device)

    def forward(self, input_ids, attention_mask):
        return self.mlm(input_ids=input_ids, attention_mask=attention_mask).logits


generator = PDGAN_Generator()
g_optimizer = optim.Adam(generator.parameters(), lr=3e-5)

print("\nStarting Stable PDGAN Training...\n")

for epoch in range(20):
    generator.train()
    # email_model stays in eval but embedding weight grad is enabled for backprop
    total_g_loss = 0.0
    num_batches = 0

    for i in tqdm(range(0, len(test_df), 6), desc=f"Epoch {epoch + 1}/20"):
        batch = test_df.iloc[i:i + 6]
        texts = batch['email_text'].tolist()

        inputs = email_tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # --- Masking ---
        mask_prob = 0.25
        mask = (torch.rand_like(input_ids, dtype=torch.float32) < mask_prob)
        mask = mask & (input_ids != email_tokenizer.pad_token_id)

        input_ids_masked = input_ids.clone()
        input_ids_masked[mask] = email_tokenizer.mask_token_id

        # --- Generator forward: logits over vocab ---
        gen_logits = generator(input_ids_masked, attention_mask)  # [B, T, vocab]

        # --- Gumbel-Softmax (straight-through, temperature-annealed) ---
        temperature = max(0.9 - epoch * 0.03, 0.3)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(gen_logits) + 1e-9) + 1e-9)
        soft_probs = torch.softmax((gen_logits + gumbel_noise) / temperature, dim=-1)  # [B, T, vocab]

        # For unmasked positions, replace soft probs with one-hot of the original token
        # so the discriminator sees real tokens for non-perturbed positions.
        one_hot_real = torch.zeros_like(soft_probs).scatter_(
            -1, input_ids.unsqueeze(-1), 1.0
        )  # [B, T, vocab]
        mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]
        # Masked positions use soft Gumbel probs; unmasked use hard one-hot
        combined_embeds = mask_expanded * soft_probs + (1 - mask_expanded) * one_hot_real
        # combined_embeds: [B, T, vocab] — differentiable w.r.t. gen_logits ✓

        # --- Discriminator: soft embedding path (no decode/re-tokenize) ---
        e_feat = email_model.forward_soft(combined_embeds, attention_mask)  # grad_fn ✓

        u_in = url_tokenizer(
            batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        # URL features are detached (URL not perturbed by generator)
        with torch.no_grad():
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)

        d_output = comm_model(e_feat, u_feat)
        d_prob = torch.sigmoid(d_output)

        # Generator loss: fool discriminator into outputting high phishing probability
        g_loss = -torch.mean(torch.log(d_prob + 1e-8))

        g_optimizer.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        g_optimizer.step()

        total_g_loss += g_loss.item()
        num_batches += 1

    avg_loss = total_g_loss / max(num_batches, 1)
    print(f"Epoch {epoch + 1} | G Loss: {avg_loss:.4f}")

torch.save(generator.state_dict(), "strong_pdgan_generator.pth")
print("\n✅ PDGAN Training Completed!")
