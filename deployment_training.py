import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import warnings
from tqdm import tqdm
import gc
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
    precision_score, recall_score, confusion_matrix, log_loss,
    brier_score_loss, roc_curve, precision_recall_curve,
    average_precision_score, cohen_kappa_score, classification_report
)
from sklearn.calibration import calibration_curve
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PLOT_DIR = "plots_final_emails_urls"
os.makedirs(PLOT_DIR, exist_ok=True)

PALETTE = {
    "train":  "#4C72B0",
    "val":    "#DD8452",
    "test":   "#55A868",
    "phish":  "#C44E52",
    "legit":  "#4C72B0",
}
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

print("Loading final_emails.csv ...")
email_df = pd.read_csv("final_emails.csv")
email_df['email_text'] = email_df.apply(
    lambda row: f"Sender: {row.get('sender', '')}\n"
                f"Subject: {row.get('subject', '')}\n\n"
                f"{row.get('email body', row.get('body', ''))}",
    axis=1
)
email_df = email_df[['email_text', 'label']].copy()
email_df = email_df[email_df['label'].isin([0, 1])].reset_index(drop=True)

print("Loading final_urls.csv ...")
url_df = pd.read_csv("final_urls.csv")
url_df = url_df[url_df['label'].isin([0, 1])].reset_index(drop=True)

print(f"Email Dataset : {len(email_df)} samples")
print(f"URL Dataset   : {len(url_df)} samples")


def plot_dataset_distributions(email_df, url_df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, df, title in zip(axes, [email_df, url_df], ["Email Dataset", "URL Dataset"]):
        counts = df['label'].value_counts().sort_index()
        bars = ax.bar(
            ["Legitimate (0)", "Phishing (1)"],
            counts.values,
            color=[PALETTE["legit"], PALETTE["phish"]],
            edgecolor="white", linewidth=0.8, width=0.5
        )
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
                    f"{v:,}\n({v/len(df)*100:.1f}%)", ha='center', va='bottom', fontsize=10)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel("Sample Count")
        ax.set_ylim(0, counts.max() * 1.18)
    plt.suptitle("Class Distribution — Input Datasets", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/01_class_distribution.png", bbox_inches='tight')
    plt.close()
    print(f"  Saved: 01_class_distribution.png")

plot_dataset_distributions(email_df, url_df)


min_size = min(len(email_df), len(url_df))
email_sample = email_df.sample(n=min_size, random_state=42).reset_index(drop=True)
url_sample   = url_df.sample(n=min_size, random_state=42).reset_index(drop=True)

paired_df = pd.DataFrame({
    'email_text': email_sample['email_text'],
    'url':        url_sample['url'],
    'label':      email_sample['label']
})

train_df, temp_df = train_test_split(paired_df, test_size=0.3, stratify=paired_df['label'], random_state=42)
val_df,   test_df = train_test_split(temp_df,   test_size=0.5, stratify=temp_df['label'],   random_state=42)

print(f"Train size     : {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size      : {len(test_df)}")

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer   = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")


class EmailAgent(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(768, 128)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(self.dropout(out.last_hidden_state[:, 0, :]))


class URLAgent(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.model = AutoModel.from_pretrained("amahdaouy/DomURLs_BERT")
        self.fc = nn.Linear(768, 128)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(self.dropout(out.last_hidden_state[:, 0, :]))


class MessagePassing(nn.Module):
    def __init__(self, dim=128, rounds=2):
        super().__init__()
        self.rounds = rounds
        self.update_e = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(0.3))
        self.update_u = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(0.3))
        self.fc = nn.Linear(dim * 2, 1)

    def forward(self, e, u):
        for _ in range(self.rounds):
            e_new = self.update_e(torch.cat([e, u], dim=1))
            u_new = self.update_u(torch.cat([u, e], dim=1))
            e, u = e_new, u_new
        return self.fc(torch.cat([e, u], dim=1)).squeeze(-1)



def compute_all_metrics(y_true, y_pred, y_prob, split=""):
    """Returns a dict of 20 metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tn + fp + fn + tp

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fpr         = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr         = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fdr         = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    prevalence  = (tp + fn) / total

    metrics = {
        "Accuracy":          accuracy_score(y_true, y_pred),
        "Precision":         precision_score(y_true, y_pred, zero_division=0),
        "Recall (Sens.)":    recall_score(y_true, y_pred, zero_division=0),
        "Specificity":       specificity,
        "F1-Score":          f1_score(y_true, y_pred, zero_division=0),
        "MCC":               matthews_corrcoef(y_true, y_pred),
        "Cohen's Kappa":     cohen_kappa_score(y_true, y_pred),
        "ROC-AUC":           roc_auc_score(y_true, y_prob),
        "Avg Precision":     average_precision_score(y_true, y_prob),
        "Log Loss":          log_loss(y_true, y_prob),
        "Brier Score":       brier_score_loss(y_true, y_prob),
        "NPV":               npv,
        "FPR":               fpr,
        "FNR":               fnr,
        "FDR":               fdr,
        "TP":                int(tp),
        "TN":                int(tn),
        "FP":                int(fp),
        "FN":                int(fn),
        "Prevalence":        prevalence,
    }
    if split:
        print(f"\n{'='*60}")
        print(f"  {split.upper()} METRICS")
        print(f"{'='*60}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:<20}: {v:.4f}")
            else:
                print(f"  {k:<20}: {v}")
    return metrics



def train_model(train_df, val_df, test_df, max_epochs=50, patience=5, batch_size=8):
    email_model = EmailAgent().to(device)
    url_model   = URLAgent().to(device)
    comm_model  = MessagePassing(rounds=2).to(device)

    optimizer = optim.AdamW(
        list(email_model.parameters()) +
        list(url_model.parameters()) +
        list(comm_model.parameters()),
        lr=8e-6, weight_decay=0.08
    )
    criterion = nn.BCEWithLogitsLoss()

    # History containers
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [],  "val_acc": [],
        "train_f1":  [],  "val_f1":  [],
        "train_mcc": [],  "val_mcc": [],
        "train_auc": [],  "val_auc": [],
    }

    best_val_loss   = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
   
        email_model.train(); url_model.train(); comm_model.train()
        tr_loss = 0.0
        tr_true, tr_pred, tr_prob = [], [], []

        for i in tqdm(range(0, len(train_df), batch_size),
                      desc=f"Epoch {epoch+1:02d}", leave=False):
            batch  = train_df.iloc[i:i + batch_size]
            labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

            e_in = email_tokenizer(batch['email_text'].tolist(), padding=True,
                                   truncation=True, max_length=256,
                                   return_tensors="pt").to(device)
            u_in = url_tokenizer(batch['url'].tolist(), padding=True,
                                 truncation=True, max_length=128,
                                 return_tensors="pt").to(device)

            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids,   u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(email_model.parameters()) +
                list(url_model.parameters()) +
                list(comm_model.parameters()), 1.0)
            optimizer.step()
            tr_loss += loss.item()

            with torch.no_grad():
                prob = torch.sigmoid(logits).cpu().numpy()
                pred = (prob > 0.5).astype(int)
            tr_true.extend(batch['label'].values)
            tr_pred.extend(pred)
            tr_prob.extend(prob)

        avg_tr_loss = tr_loss / (len(train_df) // batch_size + 1)

        email_model.eval(); url_model.eval(); comm_model.eval()
        vl_loss = 0.0
        vl_true, vl_pred, vl_prob = [], [], []

        with torch.no_grad():
            for i in range(0, len(val_df), batch_size):
                batch  = val_df.iloc[i:i + batch_size]
                labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

                e_in = email_tokenizer(batch['email_text'].tolist(), padding=True,
                                       truncation=True, max_length=256,
                                       return_tensors="pt").to(device)
                u_in = url_tokenizer(batch['url'].tolist(), padding=True,
                                     truncation=True, max_length=128,
                                     return_tensors="pt").to(device)

                e_feat = email_model(e_in.input_ids, e_in.attention_mask)
                u_feat = url_model(u_in.input_ids,   u_in.attention_mask)
                logits = comm_model(e_feat, u_feat)
                loss   = criterion(logits, labels)
                vl_loss += loss.item()

                prob = torch.sigmoid(logits).cpu().numpy()
                pred = (prob > 0.5).astype(int)
                vl_true.extend(batch['label'].values)
                vl_pred.extend(pred)
                vl_prob.extend(prob)

        avg_vl_loss = vl_loss / (len(val_df) // batch_size + 1)

        history["train_loss"].append(avg_tr_loss)
        history["val_loss"].append(avg_vl_loss)
        history["train_acc"].append(accuracy_score(tr_true, tr_pred))
        history["val_acc"].append(accuracy_score(vl_true, vl_pred))
        history["train_f1"].append(f1_score(tr_true, tr_pred, zero_division=0))
        history["val_f1"].append(f1_score(vl_true, vl_pred, zero_division=0))
        history["train_mcc"].append(matthews_corrcoef(tr_true, tr_pred))
        history["val_mcc"].append(matthews_corrcoef(vl_true, vl_pred))
        history["train_auc"].append(roc_auc_score(tr_true, tr_prob))
        history["val_auc"].append(roc_auc_score(vl_true, vl_prob))

        print(f"Epoch {epoch+1:2d} | "
              f"Train Loss: {avg_tr_loss:.4f}  Acc: {history['train_acc'][-1]:.4f} | "
              f"Val Loss: {avg_vl_loss:.4f}  Acc: {history['val_acc'][-1]:.4f}")

        
        if avg_vl_loss < best_val_loss:
            best_val_loss = avg_vl_loss
            patience_counter = 0
            best_model_state = {
                'email_model': {k: v.cpu().clone() for k, v in email_model.state_dict().items()},
                'url_model':   {k: v.cpu().clone() for k, v in url_model.state_dict().items()},
                'comm_model':  {k: v.cpu().clone() for k, v in comm_model.state_dict().items()},
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best weights
    email_model.load_state_dict({k: v.to(device) for k, v in best_model_state['email_model'].items()})
    url_model.load_state_dict({k: v.to(device) for k, v in best_model_state['url_model'].items()})
    comm_model.load_state_dict({k: v.to(device) for k, v in best_model_state['comm_model'].items()})


    email_model.eval(); url_model.eval(); comm_model.eval()

    def evaluate(df_eval):
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for i in range(0, len(df_eval), batch_size):
                batch  = df_eval.iloc[i:i + batch_size]
                e_in   = email_tokenizer(batch['email_text'].tolist(), padding=True,
                                         truncation=True, max_length=256,
                                         return_tensors="pt").to(device)
                u_in   = url_tokenizer(batch['url'].tolist(), padding=True,
                                       truncation=True, max_length=128,
                                       return_tensors="pt").to(device)
                e_feat = email_model(e_in.input_ids, e_in.attention_mask)
                u_feat = url_model(u_in.input_ids,   u_in.attention_mask)
                logits = comm_model(e_feat, u_feat)
                prob   = torch.sigmoid(logits).cpu().numpy()
                pred   = (prob > 0.5).astype(int)
                y_true.extend(batch['label'].values)
                y_pred.extend(pred)
                y_prob.extend(prob)
        return np.array(y_true), np.array(y_pred), np.array(y_prob)

    tr_true, tr_pred, tr_prob = evaluate(train_df)
    vl_true, vl_pred, vl_prob = evaluate(val_df)
    te_true, te_pred, te_prob = evaluate(test_df)

    train_metrics = compute_all_metrics(tr_true, tr_pred, tr_prob, split="Train")
    val_metrics   = compute_all_metrics(vl_true, vl_pred, vl_prob, split="Validation")
    test_metrics  = compute_all_metrics(te_true, te_pred, te_prob, split="Test")


    epochs_ran = list(range(1, len(history["train_loss"]) + 1))

  
    def plot_loss_curve():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs_ran, history["train_loss"], color=PALETTE["train"],
                lw=2, marker='o', markersize=4, label="Train Loss")
        ax.plot(epochs_ran, history["val_loss"], color=PALETTE["val"],
                lw=2, marker='s', markersize=4, label="Validation Loss")
        best_ep = int(np.argmin(history["val_loss"])) + 1
        ax.axvline(best_ep, color='grey', linestyle='--', lw=1.2,
                   label=f"Best epoch ({best_ep})")
        ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
        ax.set_title("Training & Validation Loss", fontweight='bold')
        ax.legend(); plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/02_loss_curve.png", bbox_inches='tight')
        plt.close()
        print("  Saved: 02_loss_curve.png")


    def plot_epoch_metrics():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        specs = [
            ("Accuracy",  "train_acc",  "val_acc"),
            ("F1-Score",  "train_f1",   "val_f1"),
            ("MCC",       "train_mcc",  "val_mcc"),
            ("ROC-AUC",   "train_auc",  "val_auc"),
        ]
        for ax, (title, tr_key, vl_key) in zip(axes.flat, specs):
            ax.plot(epochs_ran, history[tr_key], color=PALETTE["train"],
                    lw=2, marker='o', markersize=3, label="Train")
            ax.plot(epochs_ran, history[vl_key], color=PALETTE["val"],
                    lw=2, marker='s', markersize=3, label="Validation")
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel("Epoch"); ax.set_ylabel(title)
            ax.legend(fontsize=9)
        plt.suptitle("Per-Epoch Metrics — Train vs Validation", fontsize=14,
                     fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/03_epoch_metrics.png", bbox_inches='tight')
        plt.close()
        print("  Saved: 03_epoch_metrics.png")

    def plot_confusion_matrices():
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        sets = [
            ("Train",      tr_true, tr_pred, PALETTE["train"]),
            ("Validation", vl_true, vl_pred, PALETTE["val"]),
            ("Test",       te_true, te_pred, PALETTE["test"]),
        ]
        for ax, (name, yt, yp, color) in zip(axes, sets):
            cm = confusion_matrix(yt, yp)
            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
            annot = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
                               for row_v, row_p in zip(cm, cm_pct)])
            sns.heatmap(cm, annot=annot, fmt='', ax=ax,
                        cmap=sns.light_palette(color, as_cmap=True),
                        linewidths=0.5, linecolor='white',
                        xticklabels=["Legit (0)", "Phish (1)"],
                        yticklabels=["Legit (0)", "Phish (1)"],
                        cbar=False)
            ax.set_title(f"{name} Set", fontweight='bold')
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.suptitle("Confusion Matrices", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/04_confusion_matrices.png", bbox_inches='tight')
        plt.close()
        print("  Saved: 04_confusion_matrices.png")


    def plot_roc_curves():
        fig, ax = plt.subplots(figsize=(7, 6))
        for label, yt, yp, color in [
            ("Train",      tr_true, tr_prob, PALETTE["train"]),
            ("Validation", vl_true, vl_prob, PALETTE["val"]),
            ("Test",       te_true, te_prob, PALETTE["test"]),
        ]:
            fpr_r, tpr_r, _ = roc_curve(yt, yp)
            auc = roc_auc_score(yt, yp)
            ax.plot(fpr_r, tpr_r, color=color, lw=2,
                    label=f"{label} (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label="Random")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Train / Val / Test", fontweight='bold')
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/05_roc_curves.png", bbox_inches='tight')
        plt.close()
        print("  Saved: 05_roc_curves.png")


    def plot_pr_curves():
        fig, ax = plt.subplots(figsize=(7, 6))
        for label, yt, yp, color in [
            ("Train",      tr_true, tr_prob, PALETTE["train"]),
            ("Validation", vl_true, vl_prob, PALETTE["val"]),
            ("Test",       te_true, te_prob, PALETTE["test"]),
        ]:
            prec, rec, _ = precision_recall_curve(yt, yp)
            ap = average_precision_score(yt, yp)
            ax.plot(rec, prec, color=color, lw=2,
                    label=f"{label} (AP = {ap:.4f})")
        baseline = sum(te_true) / len(te_true)
        ax.axhline(baseline, color='grey', linestyle='--', lw=1,
                   label=f"Baseline (prevalence = {baseline:.2f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves", fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/06_pr_curves.png", bbox_inches='tight')
        plt.close()
        print("  Saved: 06_pr_curves.png")


    def plot_score_distributions():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
        for ax, (name, yt, yp) in zip(axes, [
            ("Train",      tr_true, tr_prob),
            ("Validation", vl_true, vl_prob),
            ("Test",       te_true, te_prob),
        ]):
            for cls, lbl, color in [(0, "Legitimate", PALETTE["legit"]),
                                     (1, "Phishing",   PALETTE["phish"])]:
                mask = (yt == cls)
                ax.hist(yp[mask], bins=30, alpha=0.6, color=color,
                        label=f"{lbl} (n={mask.sum()})", edgecolor='white', linewidth=0.4)
            ax.axvline(0.5, color='black', linestyle='--', lw=1, label="Threshold = 0.5")
            ax.set_title(f"{name} Set", fontweight='bold')
            ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Count")
            ax.legend(fontsize=8)
        plt.suptitle("Predicted Probability Distributions by Class",
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/07_score_distributions.png", bbox_inches='tight')
        plt.close()
        print("  Saved: 07_score_distributions.png")


    def plot_threshold_sweep():
        thresholds = np.linspace(0.01, 0.99, 200)
        metrics_sweep = {"F1": [], "MCC": [], "Precision": [], "Recall": []}
        for t in thresholds:
            pred_t = (te_prob >= t).astype(int)
            metrics_sweep["F1"].append(f1_score(te_true, pred_t, zero_division=0))
            metrics_sweep["MCC"].append(matthews_corrcoef(te_true, pred_t))
            metrics_sweep["Precision"].append(precision_score(te_true, pred_t, zero_division=0))
            metrics_sweep["Recall"].append(recall_score(te_true, pred_t, zero_division=0))

        best_f1_idx  = int(np.argmax(metrics_sweep["F1"]))
        best_mcc_idx = int(np.argmax(metrics_sweep["MCC"]))

        fig, ax = plt.subplots(figsize=(9, 5))
        colors_m = {"F1": "#4C72B0", "MCC": "#DD8452",
                    "Precision": "#55A868", "Recall": "#C44E52"}
        for m, vals in metrics_sweep.items():
            ax.plot(thresholds, vals, lw=2, label=m, color=colors_m[m])
        ax.axvline(thresholds[best_f1_idx],  color="#4C72B0", linestyle='--', lw=1.2,
                   label=f"Best F1 @ {thresholds[best_f1_idx]:.2f}")
        ax.axvline(thresholds[best_mcc_idx], color="#DD8452", linestyle='--', lw=1.2,
                   label=f"Best MCC @ {thresholds[best_mcc_idx]:.2f}")
        ax.axvline(0.5, color='black', linestyle=':', lw=1, label="Default (0.5)")
        ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Metric Value")
        ax.set_title("Threshold Sweep — Test Set", fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/08_threshold_sweep.png", bbox_inches='tight')
        plt.close()
        print("  Saved: 08_threshold_sweep.png")

        # Return best thresholds for optional use
        return float(thresholds[best_f1_idx]), float(thresholds[best_mcc_idx])

-
    def plot_calibration():
        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label="Perfect Calibration")
        for label, yt, yp, color in [
            ("Train",      tr_true, tr_prob, PALETTE["train"]),
            ("Validation", vl_true, vl_prob, PALETTE["val"]),
            ("Test",       te_true, te_prob, PALETTE["test"]),
        ]:
            frac_pos, mean_pred = calibration_curve(yt, yp, n_bins=10, strategy='uniform')
            ax.plot(mean_pred, frac_pos, color=color, lw=2, marker='o',
                    markersize=5, label=label)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Reliability / Calibration Curve", fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/09_calibration_curve.png", bbox_inches='tight')
        plt.close()
        print("  Saved: 09_calibration_curve.png")


    def plot_metrics_heatmap():
        display_metrics = [
            "Accuracy", "Precision", "Recall (Sens.)", "Specificity",
            "F1-Score", "MCC", "Cohen's Kappa", "ROC-AUC",
            "Avg Precision", "Log Loss", "Brier Score", "NPV", "FPR", "FNR",
        ]
        rows = {
            "Train":      [train_metrics[m] for m in display_metrics],
            "Validation": [val_metrics[m]   for m in display_metrics],
            "Test":       [test_metrics[m]  for m in display_metrics],
        }
        df_heat = pd.DataFrame(rows, index=display_metrics).T

        fig, ax = plt.subplots(figsize=(14, 3.5))
        sns.heatmap(df_heat, annot=True, fmt=".4f", ax=ax,
                    cmap="RdYlGn", linewidths=0.4, linecolor='white',
                    vmin=0, vmax=1, cbar_kws={"label": "Metric Value"})
        ax.set_title("Comprehensive Metrics Summary", fontsize=14, fontweight='bold', pad=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right')
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/10_metrics_heatmap.png", bbox_inches='tight')
        plt.close()
        print("  Saved: 10_metrics_heatmap.png")


    def plot_test_bar():
        display_metrics = [
            "Accuracy", "Precision", "Recall (Sens.)", "Specificity",
            "F1-Score", "MCC", "Cohen's Kappa", "ROC-AUC",
            "Avg Precision", "NPV",
        ]
        values = [test_metrics[m] for m in display_metrics]
        colors = ["#2ecc71" if v >= 0.9 else "#f39c12" if v >= 0.75 else "#e74c3c"
                  for v in values]

        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(display_metrics, values, color=colors, edgecolor='white', linewidth=0.8)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008, f"{v:.4f}",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax.set_ylim(0, 1.12)
        ax.axhline(0.9, color='green', linestyle='--', lw=1, alpha=0.4, label="0.90 line")
        ax.set_ylabel("Score"); ax.set_title("Test Set — All Metrics at a Glance",
                                              fontweight='bold', fontsize=13)
        ax.set_xticklabels(display_metrics, rotation=25, ha='right')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/11_test_metrics_bar.png", bbox_inches='tight')
        plt.close()
        print("  Saved: 11_test_metrics_bar.png")

    print("\n--- Generating plots ---")
    plot_loss_curve()
    plot_epoch_metrics()
    plot_confusion_matrices()
    plot_roc_curves()
    plot_pr_curves()
    plot_score_distributions()
    best_f1_thresh, best_mcc_thresh = plot_threshold_sweep()
    plot_calibration()
    plot_metrics_heatmap()
    plot_test_bar()


    all_results = {
        "train":      {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                       for k, v in train_metrics.items()},
        "validation": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                       for k, v in val_metrics.items()},
        "test":       {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                       for k, v in test_metrics.items()},
        "best_threshold_f1":  best_f1_thresh,
        "best_threshold_mcc": best_mcc_thresh,
        "epochs_ran": len(epochs_ran),
    }
    with open(f"{PLOT_DIR}/metrics_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: metrics_summary.json")


    with open(f"{PLOT_DIR}/classification_reports.txt", "w") as f:
        for name, yt, yp in [("Train", tr_true, tr_pred),
                               ("Validation", vl_true, vl_pred),
                               ("Test", te_true, te_pred)]:
            f.write(f"{'='*50}\n{name} Set\n{'='*50}\n")
            f.write(classification_report(yt, yp,
                    target_names=["Legitimate", "Phishing"]))
            f.write("\n")
    print(f"  Saved: classification_reports.txt")

    print(f"\n✅ All plots saved to: {PLOT_DIR}/")

    return email_model, url_model, comm_model, test_metrics



print("\nStarting training on final_emails + final_urls...")
email_model, url_model, comm_model, test_metrics = train_model(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    max_epochs=50,
    patience=5,
    batch_size=8
)


save_dir = "retrained_final_emails_urls_model"
os.makedirs(save_dir, exist_ok=True)
torch.save(email_model.state_dict(), f"{save_dir}/email_model.pth")
torch.save(url_model.state_dict(),   f"{save_dir}/url_model.pth")
torch.save(comm_model.state_dict(),  f"{save_dir}/comm_model.pth")
print(f"\n✅ Model saved to: {save_dir}/")
