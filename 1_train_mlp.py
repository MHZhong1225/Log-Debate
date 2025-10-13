import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
import gc  # 用于内存清理
from collections import Counter

# 这个新版 Dataset 的 __getitem__ 会返回 ((log_vec, ctx_vec), label)
from dataset_log_with_context_memory_optimized import LogWithContextDatasetMemoryOptimized as LogWithContextDataset
from context_agent import ContextAgent

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      default="Tbird") # BGL HDFS Tbird Hadoop
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--lr",           type=float, default=5e-4)
    return p.parse_args()

args = parse_args()

# --- 把下列常量改用 args ---
DATASET     = args.dataset
BATCH_SIZE  = args.batch_size
EPOCHS      = args.epochs
LR          = args.lr
MODEL_PATH  = f"model/{DATASET}/{DATASET}_best_mlp.pt"

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CSV = f"./datasets/{DATASET}/log_train_1p0.csv"  # 使用10%的数据集
VAL_CSV   = f"./datasets/{DATASET}/log_val_1p0.csv"
TEST_CSV  = f"./datasets/{DATASET}/log_test_1p0.csv"


# ================== 1. 定义模型 ==================
class TwoMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        # 假设 log_vec 和 ctx_vec 维度相同
        tower_input_dim = input_dim

        # Log Agent 的处理 (微观分析)
        self.log_tower = nn.Sequential(
            nn.Linear(tower_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Context Agent 的处理 (宏观分析)
        self.context_tower = nn.Sequential(
            nn.Linear(tower_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 融合后的分类头
        # 输入维度是两个模型输出维度之和
        fused_dim = (hidden_dim // 2) * 2
        self.classifier_head = nn.Linear(fused_dim, num_classes)

    def forward(self, log_vec, ctx_vec):
        
        log_features = self.log_tower(log_vec)
        ctx_features = self.context_tower(ctx_vec)
        
        # 拼接两个的输出特征
        fused_features = torch.cat([log_features, ctx_features], dim=-1)
        
        # 通过分类头得到最终的 logits
        logits = self.classifier_head(fused_features)
        
        return logits

# ================== 工具函数 ==================
def build_dataset(csv_path, split_name):
    df = pd.read_csv(csv_path)
    logs   = df["content"].tolist()
    labels = df["label"].tolist()
    ctx_agent = ContextAgent(window_size=100, hist_agg="max")
    
    # 使用新的 Dataset 类
    ds = LogWithContextDataset(
        raw_logs=logs,
        labels=labels,
        mode=split_name,
        ctx_agent=ctx_agent,
        batch_encode_size=32
    )
    return ds, np.array(labels)


def eval_multiclass(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        # 修改点：解包数据
        for (log_vec, ctx_vec), y in loader:
            log_vec = log_vec.to(device, non_blocking=True)
            ctx_vec = ctx_vec.to(device, non_blocking=True)

            # 修改点：模型接收两个输入
            logits = model(log_vec, ctx_vec)
            pred = logits.argmax(dim=1).cpu()
            y_pred.append(pred.numpy())
            y_true.append(y.cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    return f1, acc, y_true, y_pred

# ================== 构建数据集/加载器 ==================

if __name__ == "__main__":
    print(f"[Memory] Building datasets...")
    train_ds, y_train = build_dataset(TRAIN_CSV, "train")


    gc.collect()  # 清理内存
    print(f"[Memory] Train dataset built, memory cleaned")
    
    val_ds,   y_val   = build_dataset(VAL_CSV,   "val")
    gc.collect()  # 清理内存
    print(f"[Memory] Val dataset built, memory cleaned")
    
    test_ds,  y_test  = build_dataset(TEST_CSV,  "inference")
    gc.collect()  # 清理内存
    print(f"[Memory] Test dataset built, memory cleaned")

    classes = sorted(set(np.concatenate([y_train, y_val, y_test]).tolist()))
    NUM_CLASSES = int(max(classes) + 1)
    print(f"[Info] num_classes={NUM_CLASSES}, classes={classes}")



    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, # ✨ 训练时建议打乱
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ================== 构建模型 ==================
    # all-MiniLM-L6-v2 的输出维度是 384
    # ContextAgent 内部的 proj 层输出也是 hidden_dim，即 384
    INPUT_DIM = 384 
    print(f"[Info] Tower INPUT_DIM={INPUT_DIM}")

    mlp = TwoMLP(
        input_dim=INPUT_DIM,
        hidden_dim=256,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    class_counts = Counter(y_train.tolist())
    weights = torch.tensor(
        [class_counts.get(i, 0) for i in range(NUM_CLASSES)],
        dtype=torch.float
    )
    weights = 1.0 / torch.clamp(weights, min=1.0)
    weights = weights / weights.sum() * NUM_CLASSES    # 归一化到均值≈1
    weights = weights.to(DEVICE)

    print("[Info] class_counts:", class_counts)
    print("[Info] loss weights :", weights.cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=weights)    # ←⭐ 用加权损失

    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=LR)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps_per_epoch * EPOCHS   # 完整余弦退火
    )

    # ================== 训练循环 ==================
    best_f1 = 0.0
    for epoch in range(EPOCHS):
        mlp.train()
        total_loss, n_batches = 0.0, 0

        # 解包
        for (log_vec, ctx_vec), y in train_loader:
            log_vec = log_vec.to(DEVICE, non_blocking=True)
            ctx_vec = ctx_vec.to(DEVICE, non_blocking=True)
            # y = y.to(DEVICE).long().squeeze(-1)
            y = y.to(DEVICE).long()

            logits = mlp(log_vec, ctx_vec)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

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
    print(f"\n[Final Results for {DATASET}]")
    print(f"Test F1(macro): {test_f1:.6f}")
    print(f"Test Acc:       {test_acc:.6f}")
    print("Per-class report:\n", classification_report(y_true, y_pred, digits=4))