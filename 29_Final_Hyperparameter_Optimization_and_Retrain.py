# File: 29_Final_Hyperparameter_Optimization_and_Retrain_PSO_Optuna.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import warnings
from tqdm import tqdm
import optuna
from pyswarm import pso
import gc
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================== LOAD DATA ======================
print("Loading dataset...")
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset)
df = df[df['label'].isin([0, 1])].reset_index(drop=True)
df = df.rename(columns={"content": "email_text"})


def extract_first_url(text):
    import re
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else None


df['url'] = df['email_text'].apply(extract_first_url)

phishing_urls = df[df['label'] == 1]['url'].dropna().tolist()
legit_urls = df[df['label'] == 0]['url'].dropna().tolist()


def get_matching_url(label):
    if label == 1 and phishing_urls:
        return random.choice(phishing_urls)
    elif label == 0 and legit_urls:
        return random.choice(legit_urls)
    return ""


df['url'] = df.apply(lambda row: row['url'] if pd.notna(row['url']) else get_matching_url(row['label']), axis=1)

# Use 30% data for optimization to keep it fast
df_opt = df.sample(frac=0.30, random_state=42).reset_index(drop=True)
print(f"Total samples: {len(df)} | Optimization subset: {len(df_opt)}")

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")


# ====================== MODEL ======================
class EmailAgent(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(768, 128)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(pooled))


class URLAgent(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.model = AutoModel.from_pretrained("amahdaouy/DomURLs_BERT")
        self.fc = nn.Linear(768, 128)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(pooled))


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


# ====================== TRAINING FUNCTION ======================
def train_with_params(params, df_use, n_folds=5, max_epochs=30, patience=5, batch_size=8, save_model=False):
    fold_f1 = []
    best_model_state = None
    best_f1 = 0.0

    if n_folds == 1:  # Optimization Phase
        train_df, val_df = train_test_split(df_use, test_size=0.25, stratify=df_use['label'], random_state=42)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        folds = [(train_df, val_df)]
    else:  # Final Phase
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = [(df_use.iloc[train_idx].reset_index(drop=True),
                  df_use.iloc[val_idx].reset_index(drop=True))
                 for train_idx, val_idx in kf.split(df_use)]

    for fold_idx, (train_df, val_df) in enumerate(folds):
        print(f"   Fold {fold_idx + 1}/{n_folds if n_folds > 1 else '1 (split)'}")

        email_model = EmailAgent(dropout=params['dropout']).to(device)
        url_model = URLAgent(dropout=params['dropout']).to(device)
        comm_model = MessagePassing(rounds=params.get('mp_rounds', 2)).to(device)

        optimizer = optim.AdamW(
            list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()),
            lr=params['lr'], weight_decay=params['weight_decay']
        )
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs):
            email_model.train()
            url_model.train()
            comm_model.train()
            train_loss = 0.0

            for i in tqdm(range(0, len(train_df), batch_size), desc=f"Epoch {epoch + 1}", leave=False):
                batch = train_df.iloc[i:i + batch_size]
                labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)

                e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256,
                                       return_tensors="pt").to(device)
                u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128,
                                     return_tensors="pt").to(device)

                e_feat = email_model(e_in.input_ids, e_in.attention_mask)
                u_feat = url_model(u_in.input_ids, u_in.attention_mask)
                logits = comm_model(e_feat, u_feat)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()), 1.0)
                optimizer.step()
                train_loss += loss.item()

            avg_loss = train_loss / (len(train_df) // batch_size + 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

        # Validation
        email_model.eval()
        url_model.eval()
        comm_model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for i in range(0, len(val_df), batch_size):
                batch = val_df.iloc[i:i + batch_size]
                e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256,
                                       return_tensors="pt").to(device)
                u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128,
                                     return_tensors="pt").to(device)
                e_feat = email_model(e_in.input_ids, e_in.attention_mask)
                u_feat = url_model(u_in.input_ids, u_in.attention_mask)
                logits = comm_model(e_feat, u_feat)
                pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
                y_true.extend(batch['label'].values)
                y_pred.extend(pred)

        current_f1 = f1_score(y_true, y_pred)
        fold_f1.append(current_f1)

        if current_f1 > best_f1 and save_model:
            best_f1 = current_f1
            best_model_state = {
                'email_model': email_model.state_dict(),
                'url_model': url_model.state_dict(),
                'comm_model': comm_model.state_dict(),
                'params': params
            }

        del email_model, url_model, comm_model
        gc.collect()
        torch.cuda.empty_cache()

    mean_f1 = np.mean(fold_f1)

    if save_model and best_model_state:
        save_dir = "best_optimized_phishing_model"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_model_state['email_model'], f"{save_dir}/email_model.pth")
        torch.save(best_model_state['url_model'], f"{save_dir}/url_model.pth")
        torch.save(best_model_state['comm_model'], f"{save_dir}/comm_model.pth")
        torch.save(best_model_state['params'], f"{save_dir}/best_params.pth")
        print(f"\nBest model saved to: ./{save_dir}/")

    return mean_f1


# ====================== OPTIMIZATION ======================
print("=== Starting Hyperparameter Optimization (Optuna + PSO) ===")


# --- Optuna ---
def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-6, 5e-5, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "batch_size": 8,
        "mp_rounds": trial.suggest_int("mp_rounds", 1, 4),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 0.1, log=True)
    }
    return train_with_params(params, df_opt, n_folds=1, max_epochs=12, patience=4, batch_size=8)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)
optuna_params = study.best_params
optuna_score = study.best_value
print(f"Best Optuna Params: {optuna_params} (Score: {optuna_score:.4f})")


# --- PSO ---
def pso_obj(x):
    params = {
        "lr": x[0],
        "dropout": x[1],
        "mp_rounds": int(x[2]),
        "weight_decay": x[3],
        "batch_size": 8
    }
    return -train_with_params(params, df_opt, n_folds=1, max_epochs=12, patience=4, batch_size=8)


lb = [1e-6, 0.1, 1, 1e-5]
ub = [5e-5, 0.5, 4, 0.1]
xopt, fopt = pso(pso_obj, lb, ub, swarmsize=15, maxiter=12)
pso_params = {"lr": xopt[0], "dropout": xopt[1], "mp_rounds": int(xopt[2]), "weight_decay": xopt[3], "batch_size": 8}
pso_score = -fopt
print(f"Best PSO Params: {pso_params} (Score: {pso_score:.4f})")

# Select best
final_params = optuna_params if optuna_score > pso_score else pso_params
print(f"\nSelected Best Parameters: {final_params}")

# ====================== FINAL RETRAIN ======================
print("\n=== Final Full Retraining with Best Parameters (5-Fold) ===")
final_f1 = train_with_params(
    final_params,
    df,  # Full dataset
    n_folds=5,
    max_epochs=30,
    patience=5,
    batch_size=8,
    save_model=True
)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"Best Hyperparameters : {final_params}")
print(f"Final 5-Fold F1 Score : {final_f1:.4f}")
print("=" * 70)

pd.DataFrame([final_params]).to_csv("Best_Final_Parameters.csv", index=False)
print("Parameters saved → Best_Final_Parameters.csv")
print("Model saved → best_optimized_phishing_model/")
