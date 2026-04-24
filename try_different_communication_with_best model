# =============================================================================
# Stage 2: DistilBERT (Email) + DomURLs_BERT (URL) — One Communication Style
# Full version with all saving for figures, ablation, and statistical tests
# =============================================================================

import os
import json
import re
import gc
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
    classification_report
)
from sklearn.model_selection import KFold
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# CONFIG — CHANGE ONLY THIS SECTION
# =============================================================================
communication_name = "Message Passing"   # <<< CHANGE THIS LINE ONLY >>>
# Valid options:
# "No Communication", "Simple Concat", "Weighted Score",
# "Gated Fusion", "Cross Attention", "Message Passing"

MESSAGE_PASSING_ROUNDS = 2   # only used when communication_name = "Message Passing"
                              # change to 1, 2, 3, 4 for rounds ablation experiment

N_FOLDS     = 5
MAX_EPOCHS  = 30
PATIENCE    = 5
BATCH_SIZE  = 8

# Output folder — one folder per run so nothing gets overwritten
safe_name = communication_name.replace(" ", "_")
SAVE_DIR  = f"results_DistilBERT_DomURLBERT_{safe_name}_rounds{MESSAGE_PASSING_ROUNDS}"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Style        : {communication_name}")
print(f"MP Rounds    : {MESSAGE_PASSING_ROUNDS}")
print(f"Folds        : {N_FOLDS}")
print(f"Save dir     : {SAVE_DIR}")

# =============================================================================
# DATASET
# =============================================================================
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset)

label_col = 'labels' if 'labels' in df.columns else 'label'
df = df[df[label_col].isin([0, 1])].reset_index(drop=True)
df = df.rename(columns={label_col: "label", "content": "email_text"})

def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else None

df['url'] = df['email_text'].apply(extract_first_url)

phishing_urls = df[df['label'] == 1]['url'].dropna().tolist()
legit_urls    = df[df['label'] == 0]['url'].dropna().tolist()

def get_matching_url(label):
    if label == 1 and phishing_urls:
        return random.choice(phishing_urls)
    elif label == 0 and legit_urls:
        return random.choice(legit_urls)
    return ""

df['url'] = df.apply(
    lambda row: row['url'] if pd.notna(row['url']) else get_matching_url(row['label']),
    axis=1
)
print(f"Total samples after clean pairing: {len(df)}")

# =============================================================================
# MODELS — DistilBERT for email, DomURLs_BERT for URL
# =============================================================================
class DistilBertEmailAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.model   = AutoModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        pooled  = self.dropout(pooled)
        return self.fc(pooled)                          # → (batch, 128)


class DomURLBertURLAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.model   = AutoModel.from_pretrained("amahdaouy/DomURLs_BERT")
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = outputs.last_hidden_state[:, 0, :]
        pooled  = self.dropout(pooled)
        return self.fc(pooled)                          # → (batch, 128)


email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer   = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

# =============================================================================
# COMMUNICATION MODULES
# =============================================================================
class SimpleConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )
    def forward(self, e, u):
        return self.fc(torch.cat([e, u], dim=1)).squeeze(-1)


class WeightedScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.5, 0.5]))
    def forward(self, email_score, url_score):
        w = torch.softmax(self.weight, dim=0)
        return w[0] * email_score + w[1] * url_score


class GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2), nn.Softmax(dim=1)
        )
    def forward(self, e, u):
        comm  = torch.cat([e, u], dim=1)
        gate  = self.gate(comm)
        e_sc  = torch.sigmoid(torch.mean(e, dim=1))
        u_sc  = torch.sigmoid(torch.mean(u, dim=1))
        return gate[:, 0] * e_sc + gate[:, 1] * u_sc


class CrossAttention(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim   = dim
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.fc    = nn.Linear(dim * 2, 1)
    def forward(self, e, u):
        Q        = self.query(e)
        K        = self.key(u)
        V        = self.value(u)
        attn     = torch.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / (self.dim ** 0.5), dim=-1
        )
        attended = torch.matmul(attn, V)
        return self.fc(torch.cat([e, attended], dim=1)).squeeze(-1)


class MessagePassing(nn.Module):
    def __init__(self, dim=128, rounds=2):
        super().__init__()
        self.rounds   = rounds
        self.update_e = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(0.3))
        self.update_u = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(0.3))
        self.fc       = nn.Linear(dim * 2, 1)
    def forward(self, e, u):
        for _ in range(self.rounds):
            e_new = self.update_e(torch.cat([e, u], dim=1))
            u_new = self.update_u(torch.cat([u, e], dim=1))
            e, u  = e_new, u_new
        return self.fc(torch.cat([e, u], dim=1)).squeeze(-1)


class NoCommunication:
    @staticmethod
    def forward(e_feat, u_feat):
        e_score = torch.sigmoid(torch.mean(e_feat, dim=1))
        u_score = torch.sigmoid(torch.mean(u_feat, dim=1))
        return (e_score + u_score) / 2.0


comm_dict = {
    "No Communication": None,
    "Simple Concat":    SimpleConcat,
    "Weighted Score":   WeightedScore,
    "Gated Fusion":     GatedFusion,
    "Cross Attention":  CrossAttention,
    "Message Passing":  MessagePassing,
}
comm_class = comm_dict[communication_name]

# =============================================================================
# METRICS
# =============================================================================
def compute_all_metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision":         precision_score(y_true, y_pred, average='binary', zero_division=0),
        "recall":            recall_score(y_true, y_pred, average='binary', zero_division=0),
        "f1":                f1_score(y_true, y_pred, average='binary', zero_division=0),
        "f1_macro":          f1_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_weighted":       f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "mcc":               matthews_corrcoef(y_true, y_pred),
        "cohen_kappa":       cohen_kappa_score(y_true, y_pred),
        "specificity":       tn / (tn + fp) if (tn + fp) > 0 else 0,
        "confusion_matrix":  [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if y_prob is not None:
        metrics["roc_auc"]       = roc_auc_score(y_true, y_prob)
        metrics["avg_precision"] = average_precision_score(y_true, y_prob)
        metrics["log_loss"]      = log_loss(y_true, y_prob)
    return metrics

# =============================================================================
# TRAINING LOOP
# =============================================================================
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_results     = []   # aggregate metrics per fold
all_y_true       = []   # all true labels across folds (for overall metrics)
all_y_pred       = []   # all predictions across folds
all_y_prob       = []   # all probabilities across folds
per_fold_preds   = []   # per-fold predictions (for McNemar's test)
epoch_logs       = []   # training loss per epoch per fold

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    email_model = DistilBertEmailAgent().to(device)
    url_model   = DomURLBertURLAgent().to(device)

    if comm_class is MessagePassing:
        comm_model = comm_class(rounds=MESSAGE_PASSING_ROUNDS).to(device)
    elif comm_class is not None:
        comm_model = comm_class().to(device)
    else:
        comm_model = None

    params = list(email_model.parameters()) + list(url_model.parameters())
    if comm_model:
        params += list(comm_model.parameters())

    optimizer = optim.AdamW(params, lr=8e-6, weight_decay=0.08)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss    = float('inf')
    patience_counter = 0
    fold_epoch_log   = []

    # ---- Training ----
    for epoch in range(MAX_EPOCHS):
        email_model.train()
        url_model.train()
        if comm_model:
            comm_model.train()

        train_loss = 0.0
        n_batches  = 0

        progress_bar = tqdm(
            range(0, len(train_df), BATCH_SIZE),
            desc=f"Epoch {epoch + 1:2d}/{MAX_EPOCHS}",
            leave=False
        )

        for i in progress_bar:
            batch  = train_df.iloc[i:i + BATCH_SIZE]
            labels = torch.tensor(
                batch['label'].values, dtype=torch.float32
            ).to(device)

            e_in = email_tokenizer(
                batch['email_text'].tolist(),
                padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            ).to(device)

            u_in = url_tokenizer(
                batch['url'].tolist(),
                padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(device)

            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids,   u_in.attention_mask)

            if communication_name == "No Communication":
                prob   = NoCommunication.forward(e_feat, u_feat)
                logits = prob * 2 - 1
            elif communication_name == "Weighted Score":
                e_score = torch.sigmoid(torch.mean(e_feat, dim=1))
                u_score = torch.sigmoid(torch.mean(u_feat, dim=1))
                logits  = comm_model(e_score, u_score)
            else:
                logits = comm_model(e_feat, u_feat)

            loss = criterion(logits, labels)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1
            progress_bar.set_postfix({'Loss': f'{train_loss / n_batches:.4f}'})

        avg_loss = train_loss / n_batches
        fold_epoch_log.append({
            "fold":       fold + 1,
            "epoch":      epoch + 1,
            "train_loss": avg_loss
        })

        if avg_loss < best_val_loss:
            best_val_loss    = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

    epoch_logs.extend(fold_epoch_log)

    # ---- Evaluation ----
    email_model.eval()
    url_model.eval()
    if comm_model:
        comm_model.eval()

    fold_y_true = val_df['label'].values
    fold_y_pred = []
    fold_y_prob = []

    with torch.no_grad():
        for i in range(0, len(val_df), BATCH_SIZE):
            batch = val_df.iloc[i:i + BATCH_SIZE]

            e_in = email_tokenizer(
                batch['email_text'].tolist(),
                padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            ).to(device)

            u_in = url_tokenizer(
                batch['url'].tolist(),
                padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(device)

            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids,   u_in.attention_mask)

            if communication_name == "No Communication":
                prob = NoCommunication.forward(e_feat, u_feat).cpu().numpy()
            elif communication_name == "Weighted Score":
                e_score = torch.sigmoid(torch.mean(e_feat, dim=1))
                u_score = torch.sigmoid(torch.mean(u_feat, dim=1))
                prob = comm_model(e_score, u_score).cpu().numpy()
            else:
                prob = torch.sigmoid(comm_model(e_feat, u_feat)).cpu().numpy()

            pred = (prob > 0.5).astype(int)
            fold_y_pred.extend(pred.tolist())
            fold_y_prob.extend(prob.tolist())

    # Compute fold metrics
    metrics = compute_all_metrics(fold_y_true, fold_y_pred, fold_y_prob)
    fold_results.append(metrics)

    # Store per-fold predictions for McNemar's test and overall metrics
    per_fold_preds.append({
        "fold":       fold + 1,
        "val_indices": val_idx.tolist(),
        "y_true":     fold_y_true.tolist(),
        "y_pred":     fold_y_pred,
        "y_prob":     fold_y_prob
    })

    all_y_true.extend(fold_y_true.tolist())
    all_y_pred.extend(fold_y_pred)
    all_y_prob.extend(fold_y_prob)

    print(f"  Fold {fold + 1} — Acc: {metrics['accuracy']:.4f} | "
          f"F1: {metrics['f1']:.4f} | MCC: {metrics['mcc']:.4f} | "
          f"ROC-AUC: {metrics['roc_auc']:.4f}")

    del email_model, url_model
    if comm_model:
        del comm_model
    torch.cuda.empty_cache()
    gc.collect()

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================
# Average across folds (exclude non-numeric keys)
numeric_keys = [k for k in fold_results[0] if k != "confusion_matrix"]
avg_metrics  = {k: float(np.mean([f[k] for f in fold_results])) for k in numeric_keys}
std_metrics  = {k: float(np.std([f[k]  for f in fold_results])) for k in numeric_keys}

print(f"\n=== {communication_name} | DistilBERT+DomURLs_BERT | "
      f"{N_FOLDS}-fold | MP rounds={MESSAGE_PASSING_ROUNDS} ===")
for k in numeric_keys:
    print(f"  {k.replace('_', ' ').title():25s}: {avg_metrics[k]:.4f}  (±{std_metrics[k]:.4f})")

# =============================================================================
# SAVE EVERYTHING
# =============================================================================

# 1. Average metrics CSV
avg_df = pd.DataFrame([avg_metrics])
avg_df.to_csv(os.path.join(SAVE_DIR, "avg_metrics.csv"), index=False)

# 2. Per-fold metrics CSV
fold_df = pd.DataFrame([
    {k: v for k, v in f.items() if k != "confusion_matrix"}
    for f in fold_results
])
fold_df.insert(0, 'fold', range(1, N_FOLDS + 1))
fold_df.to_csv(os.path.join(SAVE_DIR, "per_fold_metrics.csv"), index=False)

# 3. Per-fold predictions, probabilities, true labels (for McNemar's test)
with open(os.path.join(SAVE_DIR, "per_fold_predictions.json"), 'w') as f:
    json.dump(per_fold_preds, f)

# 4. All predictions and probabilities concatenated (for PR/ROC curves)
np.save(os.path.join(SAVE_DIR, "all_y_true.npy"), np.array(all_y_true))
np.save(os.path.join(SAVE_DIR, "all_y_pred.npy"), np.array(all_y_pred))
np.save(os.path.join(SAVE_DIR, "all_y_prob.npy"), np.array(all_y_prob))

# 5. Confusion matrix (from all folds combined)
cm = confusion_matrix(all_y_true, all_y_pred)
np.save(os.path.join(SAVE_DIR, "confusion_matrix.npy"), cm)
print(f"\nCombined confusion matrix:\n{cm}")

# 6. Full classification report (per-class precision/recall)
report = classification_report(
    all_y_true, all_y_pred,
    target_names=["Legitimate", "Phishing"],
    output_dict=True
)
with open(os.path.join(SAVE_DIR, "classification_report.json"), 'w') as f:
    json.dump(report, f, indent=2)

print("\nClassification report (per class):")
print(classification_report(all_y_true, all_y_pred,
                             target_names=["Legitimate", "Phishing"]))

# 7. Epoch training logs (for learning curves)
epoch_log_df = pd.DataFrame(epoch_logs)
epoch_log_df.to_csv(os.path.join(SAVE_DIR, "epoch_training_log.csv"), index=False)

# 8. Full config saved for reproducibility
config = {
    "email_model":           "distilbert-base-uncased",
    "url_model":             "amahdaouy/DomURLs_BERT",
    "communication":         communication_name,
    "message_passing_rounds": MESSAGE_PASSING_ROUNDS,
    "n_folds":               N_FOLDS,
    "max_epochs":            MAX_EPOCHS,
    "patience":              PATIENCE,
    "batch_size":            BATCH_SIZE,
    "email_max_length":      256,
    "url_max_length":        128,
    "optimizer":             "AdamW",
    "lr":                    8e-6,
    "weight_decay":          0.08,
    "gradient_clip":         1.0,
    "dropout":               0.3,
    "hidden_dim":            128,
    "avg_metrics":           avg_metrics,
    "std_metrics":           std_metrics,
}
with open(os.path.join(SAVE_DIR, "config_and_results.json"), 'w') as f:
    json.dump(config, f, indent=2)

print(f"\nAll outputs saved to: {SAVE_DIR}/")
print("  avg_metrics.csv")
print("  per_fold_metrics.csv")
print("  per_fold_predictions.json   <- needed for McNemar's test")
print("  all_y_true.npy              <- needed for PR curve, ROC curve")
print("  all_y_pred.npy              <- needed for confusion matrix figure")
print("  all_y_prob.npy              <- needed for PR curve, threshold analysis")
print("  confusion_matrix.npy        <- needed for confusion matrix figure")
print("  classification_report.json  <- needed for per-class breakdown figure")
print("  epoch_training_log.csv      <- needed for training loss curve")
print("  config_and_results.json     <- full config + results for reproducibility")
