# File: LLM_Paraphrasing_Attack_Pipeline_Updated.py

import pandas as pd
import torch
import torch.nn as nn
import re
import random
import numpy as np
import gc
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoConfig
from datasets import load_dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ====================== SETUP ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("=" * 100)
print("LLM PARAPHRASING ATTACK PIPELINE - UPDATED")
print("Using saved baseline: best_trained_model.pth")
print(f"Device: {device}")
print("=" * 100)


# ====================== CLASSIFIER MODELS ======================
class DistilBertEmail(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.adapter = nn.Sequential(
            nn.Linear(768, 192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(192, 128)
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
            nn.Linear(768, 192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(192, 128)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.adapter(pooled)


class MessagePassing(nn.Module):
    def __init__(self, rounds=2):
        super().__init__()
        self.rounds = rounds
        self.update_e = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.update_u = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, e, u):
        for _ in range(self.rounds):
            e = self.update_e(torch.cat([e, u], dim=1))
            u = self.update_u(torch.cat([u, e], dim=1))
        return self.fc(torch.cat([e, u], dim=1)).squeeze(-1)


# ====================== LOAD SAVED CLASSIFIER ======================
checkpoint = torch.load("best_trained_model.pth", map_location=device)

email_model = DistilBertEmail().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing(rounds=2).to(device)

email_model.load_state_dict(checkpoint["email_model"])
url_model.load_state_dict(checkpoint["url_model"])
comm_model.load_state_dict(checkpoint["comm_model"])

email_model.eval()
url_model.eval()
comm_model.eval()

print("Best phishing classifier loaded successfully.")


# ====================== LOAD DATASET ======================
dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset)

df = df.rename(columns={
    "content": "email_text",
    "labels": "label"
})

df = df[df["label"].isin([0, 1])].reset_index(drop=True)


def extract_first_url(text):
    urls = re.findall(r"https?://\S+", str(text))
    return urls[0] if urls else ""


df["url"] = df["email_text"].apply(extract_first_url)
df = df[df["url"] != ""].reset_index(drop=True)

_, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

test_df = test_df.reset_index(drop=True)

print(f"Total usable samples with URLs: {len(df)}")
print(f"Test samples: {len(test_df)}")


# ====================== TOKENIZERS ======================
email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")


# ====================== PARAPHRASING MODELS ======================
paraphrase_models = {
    "flan_t5_large": "google/flan-t5-large",
    "pegasus_paraphrase": "tuner007/pegasus_paraphrase"
}


def load_paraphraser(model_name):
    config = AutoConfig.from_pretrained(model_name)

    if hasattr(config, "tie_word_embeddings"):
        config.tie_word_embeddings = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        config=config
    ).to(device)

    model.eval()

    return model, tokenizer


def paraphrase_with_prompt(text, model, tokenizer, style="strong"):
    text = str(text)

    if style == "strong":
        prompt = (
            "Rewrite this email to sound more professional, natural, and trustworthy "
            f"while keeping the same meaning:\n\n{text}"
        )
    else:
        prompt = f"Paraphrase the following email naturally:\n\n{text}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            early_stopping=True
        )

    paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if paraphrased.strip() == "":
        return text

    return paraphrased


# ====================== EVALUATION FUNCTION ======================
def evaluate(df_eval, use_adv=False, batch_size=8):
    y_true = []
    y_prob = []

    col = "adv_email" if use_adv else "email_text"

    with torch.no_grad():
        for i in range(0, len(df_eval), batch_size):
            batch = df_eval.iloc[i:i + batch_size]

            emails = [str(t) for t in batch[col].tolist()]
            urls = [str(u) for u in batch["url"].tolist()]

            e_in = email_tokenizer(
                emails,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)

            u_in = url_tokenizer(
                urls,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)

            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)

            logits = comm_model(e_feat, u_feat)
            prob = logits.sigmoid().detach().cpu().numpy()

            y_true.extend(batch["label"].values)
            y_prob.extend(prob)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }


# ====================== RUN CLEAN BASELINE ======================
print("\n=== STAGE 1: CLEAN BASELINE ===")

clean_metrics = evaluate(test_df, use_adv=False)

for k, v in clean_metrics.items():
    print(f"{k.replace('_', ' ').title()}: {v:.4f}")


# ====================== RUN LLM PARAPHRASING ATTACK ======================
print("\n=== STAGE 2: LLM PARAPHRASING ATTACK ===")
print("Note: This attack modifies email text only. URLs remain unchanged.")

all_results = []

for model_key, model_name in paraphrase_models.items():
    print(f"\nLoading paraphrasing model: {model_key} -> {model_name}")

    try:
        paraphraser, para_tokenizer = load_paraphraser(model_name)
    except Exception as e:
        print(f"Could not load {model_key}: {e}")
        continue

    for style in ["normal", "strong"]:
        print(f"\nAttacking with {model_key} using {style} prompt...")

        test_adv = test_df.copy()
        adv_emails = []
        failed_count = 0

        for text in tqdm(test_adv["email_text"], desc=f"{model_key}_{style}"):
            try:
                para = paraphrase_with_prompt(
                    text=text,
                    model=paraphraser,
                    tokenizer=para_tokenizer,
                    style=style
                )
                adv_emails.append(para)

            except Exception as e:
                failed_count += 1
                adv_emails.append(str(text))

        test_adv["adv_email"] = adv_emails

        output_file = f"llm_attack_outputs_{model_key}_{style}.csv"
        test_adv.to_csv(output_file, index=False)

        adv_metrics = evaluate(test_adv, use_adv=True)

        f1_drop = clean_metrics["f1"] - adv_metrics["f1"]
        acc_drop = clean_metrics["accuracy"] - adv_metrics["accuracy"]

        result = {
            "attack_model": model_key,
            "prompt_style": style,
            "clean_accuracy": clean_metrics["accuracy"],
            "clean_f1": clean_metrics["f1"],
            "adv_accuracy": adv_metrics["accuracy"],
            "adv_f1": adv_metrics["f1"],
            "adv_mcc": adv_metrics["mcc"],
            "adv_roc_auc": adv_metrics["roc_auc"],
            "accuracy_drop": acc_drop,
            "f1_drop": f1_drop,
            "failed_paraphrases": failed_count,
            "saved_file": output_file
        }

        all_results.append(result)

        print(f"Saved adversarial samples to: {output_file}")
        print(f"Failed paraphrases: {failed_count}")
        print(f"Adversarial Accuracy: {adv_metrics['accuracy']:.4f}")
        print(f"Adversarial F1: {adv_metrics['f1']:.4f}")
        print(f"F1 Drop: {f1_drop:.4f}")
        print(f"Accuracy Drop: {acc_drop:.4f}")

    del paraphraser, para_tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ====================== SAVE FINAL RESULTS ======================
results_df = pd.DataFrame(all_results)
results_df.to_csv("llm_paraphrasing_attack_summary.csv", index=False)

print("\n=== FINAL SUMMARY ===")
print(results_df)

print("\nExperiment completed.")
print("Summary saved to: llm_paraphrasing_attack_summary.csv")
