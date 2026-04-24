# File: 62_MessagePassing_Rounds_Ablation_All_Plots.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import warnings
from tqdm import tqdm
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

warnings.filterwarnings("ignore")
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*180)
print("FULL MESSAGE PASSING ABLATION + COMPREHENSIVE PLOTS")
print("1, 2, 3, 4 rounds | 5-Fold CV | All Report Visualizations")
print("="*180)

# ====================== MODEL ======================
class DistilBertEmail(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.adapter = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.5), nn.Linear(192, 128))
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
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
    def __init__(self, dim=128, rounds=2):
        super().__init__()
        self.rounds = rounds
        self.update_e = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Dropout(0.5))
        self.update_u = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Dropout(0.5))
        self.fc = nn.Linear(dim*2, 1)
    def forward(self, e, u):
        for _ in range(self.rounds):
            e = self.update_e(torch.cat([e, u], dim=1))
            u = self.update_u(torch.cat([u, e], dim=1))
        return self.fc(torch.cat([e, u], dim=1)).squeeze(-1)

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

# Load Dataset
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset).rename(columns={"content": "email_text", "labels": "label"})
df = df[df['label'].isin([0, 1])].reset_index(drop=True)

def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else ""

df['url'] = df['email_text'].apply(extract_first_url)
df = df[df['url'] != ""].reset_index(drop=True)

X = df.reset_index(drop=True)
y = df['label'].values

# ====================== EVALUATION ======================
def evaluate(email_model, url_model, comm_model, df_eval):
    y_true, y_prob = [], []
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i+8]
            emails = [str(t) for t in batch['email_text'].tolist()]
            urls = [str(u) for u in batch['url'].tolist()]

            e_in = email_tokenizer(emails, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
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
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "predictions": y_pred.tolist(),
        "probabilities": y_prob,
        "true_labels": y_true
    }

# ====================== ABLATION ======================
rounds_list = [1, 2, 3, 4]
ablation_results = {}
total_start = time.time()

for rounds in rounds_list:
    print(f"\n{'='*80}")
    print(f"RUNNING WITH {rounds} MESSAGE PASSING ROUNDS")
    print(f"{'='*80}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        train_df = X.iloc[train_idx].copy()
        val_df = X.iloc[val_idx].copy()

        email_model = DistilBertEmail().to(device)
        url_model = DomURLBERT().to(device)
        comm_model = MessagePassing(dim=128, rounds=rounds).to(device)

        for param in email_model.model.parameters(): param.requires_grad = False
        for param in url_model.model.parameters(): param.requires_grad = False

        optimizer = optim.AdamW(list(email_model.adapter.parameters()) + list(url_model.adapter.parameters()) + list(comm_model.parameters()), lr=5e-6, weight_decay=0.12)
        pos_weight = torch.tensor((1 - train_df['label'].mean()) / train_df['label'].mean()).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(10):
            email_model.train(); url_model.train(); comm_model.train()
            for i in tqdm(range(0, len(train_df), 8), desc=f"R{rounds} F{fold} E{epoch+1}", leave=False):
                batch = train_df.iloc[i:i+8]
                labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

                emails = [str(t) for t in batch['email_text'].tolist()]
                urls = [str(u) for u in batch['url'].tolist()]

                e_in = email_tokenizer(emails, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                u_in = url_tokenizer(urls, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

                e_feat = email_model(e_in.input_ids, e_in.attention_mask)
                u_feat = url_model(u_in.input_ids, u_in.attention_mask)
                logits = comm_model(e_feat, u_feat)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        metrics = evaluate(email_model, url_model, comm_model, val_df)
        fold_results.append(metrics)

    ablation_results[rounds] = fold_results

# ====================== SAVE ALL DATA ======================
with open('message_passing_ablation_full_results.json', 'w') as f:
    json.dump(ablation_results, f)

# ====================== ALL PLOTS ======================
rounds = list(ablation_results.keys())
avg_f1 = [np.mean([r['f1'] for r in ablation_results[r]]) for r in rounds]
avg_mcc = [np.mean([r['mcc'] for r in ablation_results[r]]) for r in rounds]
avg_acc = [np.mean([r['accuracy'] for r in ablation_results[r]]) for r in rounds]
avg_auc = [np.mean([r['roc_auc'] for r in ablation_results[r]]) for r in rounds]

# 1. Bar Chart - F1 Score
plt.figure(figsize=(10,6))
plt.bar([f"{r} Rounds" for r in rounds], avg_f1, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Average F1 Score vs Number of Message Passing Rounds')
plt.ylabel('Average F1 Score')
plt.grid(axis='y')
plt.savefig("Ablation_F1_Bar.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Line Plot - All Metrics
plt.figure(figsize=(10,6))
plt.plot(rounds, avg_f1, marker='o', label='F1 Score', linewidth=2.5)
plt.plot(rounds, avg_mcc, marker='s', label='MCC', linewidth=2.5)
plt.plot(rounds, avg_acc, marker='^', label='Accuracy', linewidth=2.5)
plt.plot(rounds, avg_auc, marker='d', label='ROC-AUC', linewidth=2.5)
plt.title('Performance Metrics vs Number of Message Passing Rounds')
plt.xlabel('Number of Rounds')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.savefig("Ablation_All_Metrics_Line.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Boxplot per Round (F1)
plt.figure(figsize=(10,6))
f1_data = [[r['f1'] for r in ablation_results[rounds_val]] for rounds_val in rounds]
plt.boxplot(f1_data, labels=[f"{r} Rounds" for r in rounds])
plt.title('F1 Score Distribution per Number of Rounds')
plt.ylabel('F1 Score')
plt.grid(True)
plt.savefig("Ablation_F1_Boxplot.png", dpi=300)
plt.close()

# 4. Heatmap of Metrics
metrics_df = pd.DataFrame({
    'Rounds': [f"{r} Rounds" for r in rounds],
    'Accuracy': avg_acc,
    'F1': avg_f1,
    'MCC': avg_mcc,
    'ROC-AUC': avg_auc
}).set_index('Rounds')
plt.figure(figsize=(8,5))
sns.heatmap(metrics_df, annot=True, cmap="YlGnBu", fmt=".4f")
plt.title('Performance Heatmap Across Rounds')
plt.savefig("Ablation_Metrics_Heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Confusion Matrix for Best Round (assuming 2 rounds is best)
best_round = 2
cm = confusion_matrix(ablation_results[best_round][0]['true_labels'], ablation_results[best_round][0]['predictions'])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_round} Rounds (Example Fold)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig("Best_Round_Confusion_Matrix.png", dpi=300)
plt.close()

print("\n✅ All plots generated successfully!")
print("Saved files:")
print("• Ablation_F1_Bar.png")
print("• Ablation_All_Metrics_Line.png")
print("• Ablation_F1_Boxplot.png")
print("• Ablation_Metrics_Heatmap.png")
print("• Best_Round_Confusion_Matrix.png")
print("• message_passing_ablation_full_results.json")
print("• All .npy prediction/probability files")

print(f"\nTotal time: {(time.time() - total_start)/60:.2f} minutes")
