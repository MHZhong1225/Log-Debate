import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report

from dataset_log_with_context import LogWithContextDataset
from context_agent import ContextAgent

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      default="Hadoop")
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--use_context",  action="store_true")   # 有就 True，没写就 False
    return p.parse_args()

args = parse_args()

# --- 把下列常量改用 args ---
DATASET     = args.dataset
BATCH_SIZE  = args.batch_size
EPOCHS      = args.epochs
LR          = args.lr
USE_CONTEXT = args.use_context
MODEL_PATH  = f"{DATASET}_best_mlp_mc.pt"

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CSV = f"./datasets/{DATASET}/log_train_1p0.csv"
VAL_CSV   = f"./datasets/{DATASET}/log_val.csv"
TEST_CSV  = f"./datasets/{DATASET}/log_test.csv"

# ================== 工具函数 ==================
def build_dataset(csv_path, split_name, use_context):
    df = pd.read_csv(csv_path)
    # bad = df[df['content'].apply(lambda x: not isinstance(x, str))]
    # print(bad.head())

    logs   = df["content"].tolist()
    labels = df["label"].tolist()
    # 各 split 单独的 ContextAgent，避免窗口泄漏
    ctx_agent = ContextAgent(window_size=100, hist_agg="max") # max attention mean
    ds = LogWithContextDataset(
        raw_logs=logs,
        labels=labels,
        mode=split_name,
        ctx_agent=ctx_agent,
        batch_encode_size=32,
        use_context=use_context
    )
    return ds, np.array(labels)

def infer_input_dim(dataset):
    with torch.no_grad():
        x0, _ = dataset[0]
        return int(x0.numel())

def eval_multiclass(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)                 # [B, C]
            pred = logits.argmax(dim=1).cpu() # [B]
            y_pred.append(pred.numpy())
            y_true.append(y.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    return f1, acc, y_true, y_pred

# ================== 构建数据集/加载器 ==================
train_ds, y_train = build_dataset(TRAIN_CSV, "train", USE_CONTEXT)
val_ds,   y_val   = build_dataset(VAL_CSV,   "val",   USE_CONTEXT)
test_ds,  y_test  = build_dataset(TEST_CSV,  "inference", USE_CONTEXT)

# 自动推断类别数（假设标签已是 0..K-1 的整数；若不是，先做映射）
classes = sorted(set(np.concatenate([y_train, y_val, y_test]).tolist()))
NUM_CLASSES = int(max(classes) + 1)
print(f"[Info] num_classes={NUM_CLASSES}, classes={classes}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

# ================== 构建模型（自动推断输入维度） ==================
INPUT_DIM = infer_input_dim(train_ds)
print(f"[Info] INPUT_DIM={INPUT_DIM}  (USE_CONTEXT={USE_CONTEXT})")

mlp = nn.Sequential(
    # nn.LayerNorm(INPUT_DIM),
    nn.Linear(INPUT_DIM, 512),
    nn.ReLU(),
    nn.Linear(512, 64),
    nn.ReLU(),
    nn.Linear(64, NUM_CLASSES)   # 多分类输出
).to(DEVICE)

criterion = nn.CrossEntropyLoss() # 多分类用 CE
optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)

# ================== 训练循环 ==================
best_f1 = 0.0
for epoch in range(EPOCHS):
    mlp.train()
    total_loss, n_batches = 0.0, 0

    for x, y in train_loader:
        x = x.to(DEVICE, non_blocking=True)
        # CrossEntropyLoss 要求 y 为 Long 类型的类别索引
        y = y.to(DEVICE).long().squeeze(-1)

        logits = mlp(x)                 # [B, C]
        loss = criterion(logits, y)     # y: [B]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    # 验证
    val_f1, val_acc, _, _ = eval_multiclass(mlp, val_loader, DEVICE)
    print(f"Epoch {epoch+1:03d} | Loss: {total_loss/max(1,n_batches):.4f} "
          f"| ValF1(macro): {val_f1:.4f} Acc: {val_acc:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(mlp.state_dict(), MODEL_PATH)
        print("[✓] Saved best model")

# ================== 测试 ==================
mlp.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
test_f1, test_acc, y_true, y_pred = eval_multiclass(mlp, test_loader, DEVICE)
print(f"Test F1(macro): {test_f1:.6f}")
print(f"Test Acc:       {test_acc:.6f}")
print("Per-class report:\n", classification_report(y_true, y_pred, digits=4))