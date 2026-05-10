import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel
import time
import warnings
import re
import html

warnings.filterwarnings("ignore")


# ==========================================
# 1. Configuration & Setup
# ==========================================
class Config:
    data_dir = "data"
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    sub_path = os.path.join(data_dir, "sample_submission.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = 64
    batch_size = 32
    epochs = 6
    learning_rate = 2e-5
    model_name = "./weight"  # A lightweight BERT for fast training


# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
print("Loading data...")
train_df = pd.read_csv(Config.train_path)
test_df = pd.read_csv(Config.test_path)


def clean_text(text):
    # 1. 转换为字符串并转为小写
    text = str(text).lower()

    # 2. 解码 HTML 实体 (例如把 &amp; 变成 &)
    text = html.unescape(text)

    # 3. 去除 URL (http/https/ftp 等)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 4. 去除 @用户名
    text = re.sub(r"@\w+", "", text)

    # 5. 处理哈希标签 (保留单词，去掉 #)
    text = re.sub(r"#", "", text)

    # 6. 去除换行符、制表符等
    text = re.sub(r"[\n\r\t]", " ", text)

    # 7. 去除多余的空格
    text = re.sub(r"\s+", " ", text).strip()

    text = str(text).lower()
    return text


train_df["text"] = train_df["text"].apply(clean_text)
test_df["text"] = test_df["text"].apply(clean_text)

# Train/Val Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text"].values, train_df["target"].values, test_size=0.2, random_state=42
)

# ==========================================
# 3. Model 1: Linear Model (TF-IDF + LR)
# ==========================================
print("\n--- Model: Linear Model (TF-IDF + Logistic Regression) ---")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(train_texts)
X_val_tfidf = vectorizer.transform(val_texts)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, train_labels)

lr_preds = lr_model.predict(X_val_tfidf)
lr_f1 = f1_score(val_labels, lr_preds)
print(f"Linear Model - Validation F1: {lr_f1:.4f}")

# ==========================================
# 4. PyTorch Dataset preparation
# ==========================================
print("\nPreparing Deep Learning Datasets...")
tokenizer = AutoTokenizer.from_pretrained(Config.model_name, local_files_only=True)


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer(
            text,
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


train_dataset = TweetDataset(train_texts, train_labels, tokenizer, Config.max_len)
val_dataset = TweetDataset(val_texts, val_labels, tokenizer, Config.max_len)
test_dataset = TweetDataset(test_df["text"].values, None, tokenizer, Config.max_len)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)

# ==========================================
# 5. Core Neural Network Models
# ==========================================


# 5.1 TextCNN (Analogous to LeNet/AlexNet for text via 1D convolutions)
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=2):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(
            in_channels=embed_dim, out_channels=128, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=5, padding=2
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        x = x.permute(0, 2, 1)  # [batch, embed_dim, seq_len]

        x1 = self.pool(self.relu(self.conv1(x)))  # [batch, 128, 1]
        x2 = self.pool(self.relu(self.conv2(x)))  # [batch, 128, 1]

        out = torch.cat([x1.squeeze(-1), x2.squeeze(-1)], dim=1)
        return self.fc(out)


# 5.2 BiLSTM (RNN Model)
class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=2):
        super(TextBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        out, _ = self.lstm(x)

        mask_expanded = attention_mask.unsqueeze(-1).expand(out.size()).float()
        sum_embeddings = torch.sum(out * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        return self.fc(pooled)


# 5.3 Own Design: Transformer + CNN (SOTA hybrid design)
class BERT_CNN_Hybrid(nn.Module):
    def __init__(self, model_name, num_classes=2):
        super(BERT_CNN_Hybrid, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        embed_dim = self.bert.config.hidden_size

        self.conv = nn.Conv1d(
            in_channels=embed_dim, out_channels=256, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Linear(256 + embed_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, embed_dim]
        cls_output = sequence_output[:, 0]  # Extract CLS token

        # CNN to extract local n-gram features from BERT embeddings
        x = sequence_output.permute(0, 2, 1)
        conv_out = self.pool(self.relu(self.conv(x))).squeeze(-1)

        # Concat CLS and CNN features
        combined = torch.cat([cls_output, conv_out], dim=1)
        combined = self.dropout(combined)
        return self.fc(combined)


# ==========================================
# 6. Training & Evaluation Pipeline
# ==========================================
def train_and_eval(model, train_loader, val_loader, epochs=3, lr=2e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = 0
    logs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(Config.device)
                attention_mask = batch["attention_mask"].to(Config.device)
                labels = batch["labels"].numpy()
                outputs = model(input_ids, attention_mask)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                preds.extend(batch_preds)
                true_labels.extend(labels)

        f1 = f1_score(true_labels, preds)
        avg_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Val F1: {f1:.4f}"
        )
        logs.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_f1": f1
        })
        if f1 > best_f1:
            best_f1 = f1
    return best_f1, logs


vocab_size = tokenizer.vocab_size

print("\n--- Model: TextCNN (1D Convolution architecture) ---")
cnn_model = TextCNN(vocab_size).to(Config.device)
cnn_f1, cnn_logs = train_and_eval(cnn_model, train_loader, val_loader, epochs=5, lr=1e-3)

print("\n--- Model: BiLSTM (Recurrent Neural Network) ---")
lstm_model = TextBiLSTM(vocab_size).to(Config.device)
lstm_f1, lstm_logs = train_and_eval(lstm_model, train_loader, val_loader, epochs=5, lr=1e-3)

print("\n--- Model: Hybrid SOTA Design (DistilBERT + CNN) ---")
hybrid_model = BERT_CNN_Hybrid(Config.model_name).to(Config.device)
hybrid_f1, hybrid_logs = train_and_eval(
    hybrid_model,
    train_loader,
    val_loader,
    epochs=Config.epochs,
    lr=Config.learning_rate,
)

# Save logs for visualization
all_logs = []
for model_name, logs in [("TextCNN", cnn_logs), ("BiLSTM", lstm_logs), ("Hybrid_SOTA", hybrid_logs)]:
    for log in logs:
        log["model"] = model_name
        all_logs.append(log)

log_df = pd.DataFrame(all_logs)
log_df.to_csv("training_log_solution.csv", index=False)
print("\nTraining logs saved to 'training_log_solution.csv'")

# ==========================================
# 7. Prediction & Model Comparison
# ==========================================
print("\n============== EVALUATION SUMMARY ==============")
print(f"Linear (Baseline) F1 : {lr_f1:.4f}")
print(f"TextCNN F1           : {cnn_f1:.4f}")
print(f"BiLSTM F1            : {lstm_f1:.4f}")
print(f"SOTA Hybrid F1       : {hybrid_f1:.4f}")
print("================================================")

print("\nGenerating predictions with the best model (Hybrid SOTA)...")
hybrid_model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(Config.device)
        attention_mask = batch["attention_mask"].to(Config.device)
        outputs = hybrid_model(input_ids, attention_mask)
        batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
        test_preds.extend(batch_preds)

sub_df = pd.read_csv(Config.sub_path)
sub_df["target"] = test_preds
sub_df.to_csv("final_submission.csv", index=False)
print("Saved predictions to 'final_submission.csv' successfully!")
