# File: 16_TextCNN_Email_only.py
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score, average_precision_score, log_loss
)
from transformers import AutoTokenizer
from datasets import load_dataset
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================== GPU CHECK ======================
print("="*80)
print("GPU STATUS CHECK")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("="*80)

# ====================== LOAD LARGE DATASET ======================
print("\nLoading large training dataset...")
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset)

label_col = 'labels' if 'labels' in df.columns else 'label'
df = df[df[label_col].isin([0, 1])].reset_index(drop=True)
df = df.rename(columns={label_col: "label", "content": "email_text"})

print(f"Training email samples: {len(df)}")

# ====================== TextCNN MODEL ======================
class TextCNN(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=300, num_filters=100, filter_sizes=[3,4,5]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k) 
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 128)
        
    def forward(self, x):
        x = self.embedding(x)                    # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)                    # (batch, embed_dim, seq_len)
        x = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(x, dim=1)                  # (batch, num_filters * len(filter_sizes))
        x = self.dropout(x)
        return self.fc(x)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ====================== ALL METRICS ======================
def compute_all_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='binary', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='binary', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='binary', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["avg_precision"] = average_precision_score(y_true, y_prob)
        metrics["log_loss"] = log_loss(y_true, y_prob)
    return metrics

# ====================== 10-FOLD TRAINING WITH EARLY STOPPING ======================
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"\n--- Fold {fold+1}/10 ---")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    email_model = TextCNN().to(device)

    optimizer = optim.AdamW(email_model.parameters(), lr=8e-6, weight_decay=0.08)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    max_epochs = 30

    for epoch in range(max_epochs):
        email_model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(range(0, len(train_df), 8), desc=f"Epoch {epoch+1:2d}/{max_epochs}", leave=True)

        for i in progress_bar:
            batch = train_df.iloc[i:i+8]
            labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

            email_inputs = tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            email_feat = email_model(email_inputs.input_ids)

            comm = email_feat  # Email-only

            loss = criterion(comm.mean(dim=1), labels)
            torch.nn.utils.clip_grad_norm_(email_model.parameters(), max_norm=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (comm.mean(dim=1).sigmoid() > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

            progress_bar.set_postfix({
                'Train Loss': f'{train_loss / (i//8 + 1):.4f}',
                'Train Acc ': f'{train_correct / train_total:.4f}'
            })

        avg_train_loss = train_loss / (len(train_df) // 8 + 1)

        # Early Stopping Check
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            patience_counter = 0
            print(f"    → New best loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"    → No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"    Early stopping triggered at epoch {epoch+1} (patience = {patience})")
            break

    # ====================== EVALUATION ======================
    email_model.eval()

    y_true = val_df['label'].values
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for i in range(0, len(val_df), 32):
            batch = val_df.iloc[i:i+32]

            email_inputs = tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            email_feat = email_model(email_inputs.input_ids)
            prob = email_feat.mean(dim=1).sigmoid().cpu().numpy()
            pred = (prob > 0.5).astype(int)

            y_pred.extend(pred)
            y_prob.extend(prob)

    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    fold_results.append(metrics)

    del email_model
    torch.cuda.empty_cache()
    gc.collect()

# ====================== FINAL RESULTS ======================
avg_metrics = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
print(f"\nFinal Average for TextCNN (Email-only):")
for k, v in avg_metrics.items():
    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")

result_df = pd.DataFrame([avg_metrics])
result_df.to_csv("16_TextCNN_Email_only_results.csv", index=False)
print("\nResults saved to 16_TextCNN_Email_only_results.csv")
