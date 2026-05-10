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
import time
import warnings
warnings.filterwarnings('ignore')

# Config
DATA_DIR = 'data'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
SUB_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

print("Loading Data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Basic Preprocessing
train_df['text'] = train_df['text'].str.lower()
test_df['text'] = test_df['text'].str.lower()

X_train, X_val, y_train, y_val = train_test_split(
    train_df['text'].values, train_df['target'].values, test_size=0.2, random_state=42
)

# 1. Linear Model (TF-IDF + LR)
print("\n--- Model 1: Linear (TF-IDF + Logistic Regression) ---")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)
lr_preds = lr.predict(X_val_tfidf)
lr_f1 = f1_score(y_val, lr_preds)
print(f"Linear Val F1: {lr_f1:.4f}")

# 2. TextCNN (PyTorch)
print("\n--- Model 2: TextCNN (1D Convolutions) ---")
# Simple Word Level Tokenization for CNN
class SimpleTokenizer:
    def __init__(self, texts):
        words = set()
        for t in texts:
            words.update(t.split())
        self.word2idx = {w: i+1 for i, w in enumerate(words)}
        self.word2idx['<PAD>'] = 0
        self.vocab_size = len(self.word2idx)
        
    def encode(self, text, max_len=64):
        tokens = [self.word2idx.get(w, 0) for w in text.split()][:max_len]
        return tokens + [0] * (max_len - len(tokens))

stk = SimpleTokenizer(X_train)

class CNN_Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.data = [tokenizer.encode(t, max_len) for t in texts]
        self.labels = labels
    def __len__(self): return len(self.data)
    def __getitem__(self, i): 
        return torch.tensor(self.data[i]), torch.tensor(self.labels[i])

train_ds = CNN_Dataset(X_train, y_train, stk)
val_ds = CNN_Dataset(X_val, y_val, stk)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 2)
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextCNN(stk.vocab_size).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        crit(model(xb), yb).backward()
        opt.step()
    
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            preds.extend(torch.argmax(model(xb), dim=1).cpu().tolist())
    cnn_f1 = f1_score(y_val, preds)
    print(f"Epoch {epoch+1} Val F1: {cnn_f1:.4f}")

print("\n--- Summary Comparison ---")
print(f"Logistic Regression F1: {lr_f1:.4f}")
print(f"TextCNN F1           : {cnn_f1:.4f}")

# Final Prediction with LR (fastest)
test_tfidf = vectorizer.transform(test_df['text'].values)
test_preds = lr.predict(test_tfidf)
sub_df = pd.read_csv(SUB_PATH)
sub_df['target'] = test_preds
sub_df.to_csv('baseline_submission.csv', index=False)
print("\nGenerated 'baseline_submission.csv'")
