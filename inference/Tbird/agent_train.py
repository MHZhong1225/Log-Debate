import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.utils import shuffle
import os
import numpy as np
import argparse
import random
from sentence_transformers import SentenceTransformer
# ==== 类别映射 ====
CATEGORIES = ['-', 'CHK_DSK', 'CPU', 'ECC', 'EXT_FS', 'MPT', 'NMI', 'PBS_BFD', 'PBS_CON', 'SCSI', 'VAPI']
label2id = {label: i for i, label in enumerate(CATEGORIES)}
id2label = {i: label for label, i in label2id.items()}
NUM_CLASSES = len(CATEGORIES)

# ==== 超参数 ====
BATCH_SIZE = 64
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_LEN = 256
PATIENCE = 3
BERT_NAME = 'bert-large-uncased'
DATA_PATH = './datasets/Tbird/results/debate_agent_results.csv'
# BERT_NAME = 'sentence-transformers/all-MiniLM-L6-v2' # 新的模型名称

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
args = parser.parse_args()

SAVE_DIR = args.save_dir
os.makedirs(SAVE_DIR, exist_ok=True)

# 固定随机种子函数
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DebateDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx]
        }

class BERTEncoder(nn.Module):
    def __init__(self, model_name=BERT_NAME):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden1=512, hidden2=256, output_dim=NUM_CLASSES):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch=None):
    model.train()
    total_loss, y_true, y_pred = 0, [], []
    loop = tqdm(dataloader, desc=f"[训练] Epoch {epoch}", leave=False)
    for batch in loop:
        x, y = batch['features'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        y_true.extend(y.cpu().numpy())
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
        loop.set_postfix(loss=loss.item())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return total_loss / len(dataloader), acc, f1, prec, rec

def evaluate(model, dataloader, criterion, device, epoch=None):
    model.eval()
    total_loss, y_true, y_pred = 0, [], []
    loop = tqdm(dataloader, desc=f"[验证] Epoch {epoch}", leave=False)
    with torch.no_grad():
        for batch in loop:
            x, y = batch['features'].to(device), batch['label'].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
            loop.set_postfix(loss=loss.item())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return total_loss / len(dataloader), acc, f1, prec, rec

def stratified_split(features, labels, val_ratio=0.1):
    train_feats, train_labs, val_feats, val_labs = [], [], [], []

    for class_id in torch.unique(labels):
        idx = (labels == class_id).nonzero(as_tuple=True)[0]
        idx = shuffle(idx, random_state=args.seed)

        n_val = max(1, int(len(idx) * val_ratio))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        train_feats.append(features[train_idx])
        train_labs.append(labels[train_idx])
        val_feats.append(features[val_idx])
        val_labs.append(labels[val_idx])

    train_features = torch.cat(train_feats)
    train_labels = torch.cat(train_labs)
    val_features = torch.cat(val_feats)
    val_labels = torch.cat(val_labs)

    return train_features, train_labels, val_features, val_labels


def main():
    print(f"随机种子: {args.seed}")
    print(f"结果保存目录: {SAVE_DIR}")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA_PATH)
    # df = df[['label', 'content', 'debate_decision']].dropna()
    df = df[['label', 'content']].dropna()
    df = df[df['label'].isin(CATEGORIES)].copy()

    half_labels = ['ECC', 'PBS_BFD', 'CHK_DSK', 'NMI']
    one_percent_labels = ['-', 'PBS_CON', 'VAPI', 'MPT', 'EXT_FS', 'CPU', 'SCSI']
    sampled_dfs = []

    for label in half_labels:
        sub = df[df['label'] == label]
        sampled = sub.sample(frac=0.5, random_state=args.seed)
        sampled_dfs.append(sampled)

    for label in one_percent_labels:
        sub = df[df['label'] == label]
        n = max(1, int(0.01 * len(sub)))
        sampled = sub.sample(n=n, random_state=args.seed)
        sampled_dfs.append(sampled)

    train_val_df = pd.concat(sampled_dfs).reset_index(drop=True)
    test_df = df.drop(train_val_df.index).reset_index(drop=True) #index乱序，得修改

    train_val_df['label'] = train_val_df['label'].map(label2id)
    test_df['label'] = test_df['label'].map(label2id)

    # train_val_df['combined'] = train_val_df['content'].astype(str) + ' ' + train_val_df['debate_decision'].astype(str)
    # test_df['combined'] = test_df['content'].astype(str) + ' ' + test_df['debate_decision'].astype(str)

    train_val_df['combined'] = train_val_df['content'].astype(str) 
    test_df['combined'] = test_df['content'].astype(str) 

    tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
    encoder = BERTEncoder().to(device)

    def encode_texts(texts):
        features_list = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="BERT 编码中"):
            batch_texts = texts[i:i + BATCH_SIZE]
            encodings = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            cls_vecs = encoder(input_ids, attention_mask)
            features_list.append(cls_vecs.cpu())
        return torch.cat(features_list)

    train_val_features = encode_texts(train_val_df['combined'].tolist())
    test_features = encode_texts(test_df['combined'].tolist())
    train_val_labels = torch.tensor(train_val_df['label'].tolist(), dtype=torch.long)
    test_labels = torch.tensor(test_df['label'].tolist(), dtype=torch.long)

    train_features, train_labels, val_features, val_labels = stratified_split(train_val_features, train_val_labels)

    train_dataset = DebateDataset(train_features, train_labels)
    val_dataset = DebateDataset(val_features, val_labels)
    test_dataset = DebateDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = MLPClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float('inf')
    patience_counter = PATIENCE
    history = []

    for epoch in range(EPOCHS):
        print(f"\n== 第 {epoch + 1} 轮训练 ==")
        train_loss, train_acc, train_f1, train_prec, train_rec = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(model, val_loader, criterion, device, epoch + 1)

        print(f"[训练集] Loss: {train_loss:.4f} | Acc: {train_acc:.4f}  | P: {train_prec:.4f} | R: {train_rec:.4f} | F1: {train_f1:.4f}")
        print(f"[验证集] Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | P: {val_prec:.4f} | R: {val_rec:.4f} | F1: {val_f1:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'train_precision': train_prec,
            'train_recall': train_rec,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_precision': val_prec,
            'val_recall': val_rec
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = PATIENCE
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print("验证集 Loss 未下降，提前停止训练。")
                break

    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'debate_best_model.pt'))
    pd.DataFrame(history).to_csv(os.path.join(SAVE_DIR, 'debate_training_log.csv'), index=False)

    print("\n== 测试集评估 ==")
    test_loss, test_acc, test_f1, test_prec, test_rec = evaluate(model, test_loader, criterion, device)
    print(f"[测试集] Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | P: {test_prec:.4f} | R: {test_rec:.4f} | F1: {test_f1:.4f}")

    results = {
        'split': ['train', 'val', 'test'],
        'accuracy': [train_acc, val_acc, test_acc],
        'precision': [train_prec, val_prec, test_prec],
        'recall': [train_rec, val_rec, test_rec],
        'f1_score': [train_f1, val_f1, test_f1]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(SAVE_DIR, 'final_results_summary.csv'), index=False)
    print(f"\n[结果保存] 最终评估指标已保存至 {os.path.join(SAVE_DIR, 'final_results_summary.csv')}")

if __name__ == "__main__":
    main()
