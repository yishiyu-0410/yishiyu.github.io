import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import re
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 1. Configuration
# ==========================================
class Config:
    model_name = "roberta-base"  # Stronger than DistilBERT
    max_len = 128
    batch_size = 16
    epochs = 4
    lr = 2e-5
    n_folds = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(Config.seed)


# ==========================================
# 2. Data Cleaning & Feature Engineering
# ==========================================
def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"<.*?>", "", text)  # Remove HTML
    text = re.sub(r"@\w+", "", text)  # Remove Mentions
    text = re.sub(r"#", "", text)  # Remove Hashtag symbol
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("Loading and Engineering Features...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Use Keyword info (Critical for this competition)
train_df["keyword"] = train_df["keyword"].fillna("")
test_df["keyword"] = test_df["keyword"].fillna("")

train_df["processed_text"] = (
    train_df["keyword"] + " " + train_df["text"].apply(clean_text)
)
test_df["processed_text"] = test_df["keyword"] + " " + test_df["text"].apply(clean_text)


# ==========================================
# 3. Dataset & Model
# ==========================================
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            str(self.texts[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        data = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        if self.labels is not None:
            data["labels"] = torch.tensor(self.labels[item], dtype=torch.long)
        return data


class DisasterModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        # Using a slightly complex head for better feature extraction
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, ids, mask):
        out = self.roberta(input_ids=ids, attention_mask=mask)
        # Use Mean Pooling across the sequence for more robust features than CLS alone
        pooled_output = torch.mean(out.last_hidden_state, 1)
        return self.fc(self.drop(pooled_output))


# ==========================================
# 4. Training with K-Fold & Warmup
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(Config.model_name)


def train_fold(fold, train_idx, val_idx):
    print(f"\n--- Training Fold {fold + 1} ---")
    fold_logs = []
    train_ds = TweetDataset(
        train_df.iloc[train_idx]["processed_text"].values,
        train_df.iloc[train_idx]["target"].values,
        tokenizer,
        Config.max_len,
    )
    val_ds = TweetDataset(
        train_df.iloc[val_idx]["processed_text"].values,
        train_df.iloc[val_idx]["target"].values,
        tokenizer,
        Config.max_len,
    )

    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size)

    model = DisasterModel(Config.model_name).to(Config.device)
    optimizer = AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-2)

    num_train_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
    )

    criterion = nn.CrossEntropyLoss()
    best_f1 = 0

    for epoch in range(Config.epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                batch["input_ids"].to(Config.device),
                batch["attention_mask"].to(Config.device),
            )
            loss = criterion(outputs, batch["labels"].to(Config.device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    batch["input_ids"].to(Config.device),
                    batch["attention_mask"].to(Config.device),
                )
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                targets.extend(batch["labels"].numpy())

        val_f1 = f1_score(targets, preds)
        print(f"Epoch {epoch + 1}: Val F1 = {val_f1:.4f}")
        
        fold_logs.append({
            "fold": fold + 1,
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_f1": val_f1
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"model_fold_{fold}.bin")

    return best_f1, fold_logs


# Execution
skf = StratifiedKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.seed)
fold_scores = []
all_logs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.target)):
    score, logs = train_fold(fold, train_idx, val_idx)
    fold_scores.append(score)
    all_logs.extend(logs)

# Save logs to CSV for visualization
log_df = pd.DataFrame(all_logs)
log_df.to_csv("training_log_advanced.csv", index=False)
print(f"\nTraining logs saved to 'training_log_advanced.csv'")

print(f"\nCV Mean F1 Score: {np.mean(fold_scores):.4f}")

# ==========================================
# 5. Inference (Ensemble)
# ==========================================
print("\nPerforming Ensemble Inference on Test Set...")
test_ds = TweetDataset(
    test_df["processed_text"].values, None, tokenizer, Config.max_len
)
test_loader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False)

final_preds = np.zeros((len(test_df), 2))

for fold in range(Config.n_folds):
    model = DisasterModel(Config.model_name).to(Config.device)
    model.load_state_dict(torch.load(f"model_fold_{fold}.bin"))
    model.eval()
    fold_preds = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                batch["input_ids"].to(Config.device),
                batch["attention_mask"].to(Config.device),
            )
            # Get probabilities
            fold_preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
    final_preds += np.vstack(fold_preds) / Config.n_folds

submission = pd.read_csv("data/sample_submission.csv")
submission["target"] = np.argmax(final_preds, axis=1)
submission.to_csv("advanced_submission.csv", index=False)
print("Saved cross-validated predictions to 'advanced_submission.csv'!")
