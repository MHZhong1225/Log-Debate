#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import random
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# pip install torch transformers scikit-learn pandas sentence-transformers
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer # 保留用于获取max_len等信息，但不再手动编码

# -----------------------------
# Configuration
# -----------------------------
SEED            = int(os.getenv("SEED",            "42"))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE",      "128"))
EPOCHS          = int(os.getenv("EPOCHS",          "1000"))
LR              = float(os.getenv("LR",            "3e-3"))
WEIGHT_DECAY    = float(os.getenv("WEIGHT_DECAY",  "5e-4"))
DROP_RATE       = float(os.getenv("DROP_RATE",     "0.3"))
HID             = int(os.getenv("HID",             "64"))
MAX_LEN         = int(os.getenv("MAX_LEN",         "384"))
PATIENCE        = int(os.getenv("PATIENCE",        "35"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING","0.1"))

BERT_NAME       = os.getenv("BERT_NAME",           "sentence-transformers/all-MiniLM-L6-v2")
DATA_PATH       = os.getenv("DATA_PATH", "./datasets/Hadoop/explanation/hadoop_debate.csv")
SAVE_DIR        = os.getenv("SAVE_DIR",  "./datasets/Hadoop/results")
os.makedirs(SAVE_DIR, exist_ok=True)

USE_TINY_MLP_HEAD    = True
USE_SAMPLER_BALANCE  = True
FINE_TUNE_LAST_N     = int(os.getenv("FINE_TUNE_LAST_N", "2")) # >0时开启微调

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tqdm_settings = dict(ncols=120, leave=False)


# -----------------------------
# Utilities (不变)
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def find_label_column(df: pd.DataFrame) -> str:
    for col in ["label", "y", "target", "category"]:
        if col in df.columns: return col
    raise ValueError("未找到标签列。")

def map_labels(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    classes = sorted(df[label_col].astype(str).unique())
    label2id = {c:i for i,c in enumerate(classes)}
    id2label = {i:c for c,i in label2id.items()}
    df = df.copy()
    df["label"] = df[label_col].astype(str).map(label2id)
    return df, label2id, id2label

# <--- 关键改动：删除了 BertCLSExtractor 和 encode_to_features 两个复杂的函数 --->

# -----------------------------
# Classification Heads & Dataset (不变)
# -----------------------------
class LinearHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x): return self.fc(x)

class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(p), nn.Linear(hidden, output_dim))
    def forward(self, x): return self.net(x)

class FeatureDataset(Dataset):
    def __init__(self, feats: torch.Tensor, labels: torch.Tensor):
        self.x, self.y = feats, labels
    def __len__(self): return self.x.size(0)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# -----------------------------
# Metrics / Train / Eval (不变)
# -----------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return dict(accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
                recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
                f1=f1_score(y_true, y_pred, average="macro", zero_division=0))

def run_epoch(model, loader, criterion, optimizer=None, device: torch.device = DEVICE):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, n_samples = 0.0, 0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Train" if train_mode else "Eval", **tqdm_settings)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        if train_mode: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            logits = model(xb)
            loss = criterion(logits, yb)
            if train_mode:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(yb.cpu().numpy())
    y_pred, y_true = np.concatenate(all_preds), np.concatenate(all_labels)
    metrics = compute_metrics(y_true, y_pred)
    return total_loss / max(1, n_samples), metrics

# -----------------------------
# Main Workflow
# -----------------------------
def main():
    set_seed(SEED)
    print(f"使用设备: {DEVICE}\n使用模型: {BERT_NAME}")

    # 1. 加载数据
    df = pd.read_csv(Path(DATA_PATH))

    # 2. 预处理与文本特征准备
    if "content" not in df.columns: raise ValueError('CSV必须包含 "content" 列。')
    
    # <--- 关键改动：简化文本对的准备流程 --->
    # text_a = df["content"].astype(str).fillna("").tolist()
    text_b = None
    # if "final_explanation" in df.columns:
    #     print("[INFO] 发现 'final_explanation' 列，将与 'content' 配对。")
    #     text_b = df["final_explanation"].astype(str).fillna("").tolist()
    #     # 将 text_a 和 text_b 合并成一个适合.encode()的列表
    #     # text_pairs_to_encode = [list(pair) for pair in zip(text_a, text_b)]
    #     text_pairs_to_encode = text_b
    # else:
    #     print("[WARN] 未找到 'final_explanation' 列, 模型将仅使用 'content' 列。")
    #     text_pairs_to_encode = text_a # 如果没有解释，就只编码content
    text_a = df["content"].astype(str).fillna("")
    if "final_explanation" in df.columns:
        print("[INFO] 发现 'final_explanation' 列，将与 'content' 拼接。")
        text_b = df["final_explanation"].astype(str).fillna("")
        # Use .tolist() only at the very end when converting the final pandas Series
        text_pairs_to_encode = text_b.tolist()
        # text_pairs_to_encode = (text_a + " [SEP] " + text_b).tolist()
    else:
        text_pairs_to_encode = text_a.tolist()

    label_col = find_label_column(df)
    df, label2id, id2label = map_labels(df, label_col)
    n_classes = len(label2id)
    print(f"[数据] 样本总数: {len(df)} | 类别数: {n_classes} -> {label2id}")
    label_col = find_label_column(df)
    print(f"自动识别的标签列为: '{label_col}'")
    print("\n--- Hadoop 数据集标签分布 ---")
    print(df[label_col].value_counts())
    print("-------------------------------------\n")

    # 3. 数据集划分 (比例划分)
    print("[INFO] 划分数据集...")
    train_val_indices, test_indices = train_test_split(df.index, test_size=0.955, random_state=SEED, stratify=df.label)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.12, random_state=SEED, stratify=df.label[train_val_indices])
    
    print(f"[划分] 训练集={len(train_indices)}, 验证集={len(val_indices)}, 测试集={len(test_indices)}")

    # 4. 初始化 Encoder 模型
    # <--- 关键改动：直接加载 SentenceTransformer 模型 --->
    encoder_model = SentenceTransformer(BERT_NAME, device=DEVICE)
    encoder_model.max_seq_length = MAX_LEN

    # 5. 使用 .encode() 方法直接生成所有特征向量
    # <--- 关键改动：一步到位生成所有嵌入 --->
    print("[INFO] 开始使用 model.encode() 生成文本嵌入...")
    all_embeddings = encoder_model.encode(
        text_pairs_to_encode,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True # 直接输出PyTorch Tensor
    ).float() # 确保是float32
    
    train_feats = all_embeddings[train_indices]
    val_feats   = all_embeddings[val_indices]
    test_feats  = all_embeddings[test_indices]
    
    y_train = torch.tensor(df.loc[train_indices, "label"].values, dtype=torch.long)
    y_val   = torch.tensor(df.loc[val_indices,   "label"].values, dtype=torch.long)
    y_test  = torch.tensor(df.loc[test_indices,  "label"].values, dtype=torch.long)

    # 6. 创建 Dataset 和 DataLoader
    ds_train = FeatureDataset(train_feats, y_train)
    ds_val   = FeatureDataset(val_feats, y_val)
    ds_test  = FeatureDataset(test_feats,  y_test)

    # ... (DataLoader创建部分不变) ...
    if USE_SAMPLER_BALANCE and len(ds_train) > 0:
        class_counts = np.bincount(y_train.numpy())
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = class_weights[y_train.numpy()]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

    # 7. 创建分类头模型
    in_dim = encoder_model.get_sentence_embedding_dimension()
    if USE_TINY_MLP_HEAD:
        classifier_head = TinyMLP(input_dim=in_dim, hidden=HID, output_dim=n_classes, p=DROP_RATE).to(DEVICE)
    else:
        classifier_head = LinearHead(input_dim=in_dim, output_dim=n_classes).to(DEVICE)
    print(f"分类头模型:\n{classifier_head}")

    # 8. 定义损失函数、优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # <--- 关键改动：简化优化器参数设置 --->
    if FINE_TUNE_LAST_N > 0:
        print("[INFO] 微调模式已开启，将同时训练Encoder和分类头。")
        # 同时优化 encoder_model 和 classifier_head 的参数
        params_to_optimize = list(encoder_model.parameters()) + list(classifier_head.parameters())
    else:
        print("[INFO] 微调模式已关闭，仅训练分类头。")
        encoder_model.eval() # 冻结Encoder
        params_to_optimize = classifier_head.parameters()
        
    optimizer = optim.AdamW(params_to_optimize, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5)

    # ... (训练、评估、保存的逻辑基本不变，但传递的model是分类头) ...
    # 9. 训练循环（带早停）
    best_val_loss = float("inf")
    best_state_dict = None
    patience_cnt = 0

    print("\n--- 开始训练 ---")
    # 注意：现在训练循环只针对分类头，特征已经预先计算好
    # 如果要实现端到端的微调，需要重构训练循环，这里保持原逻辑，适用于预计算特征
    
    # 将模型组合起来进行训练
    full_model = nn.Sequential(encoder_model, classifier_head).to(DEVICE) if FINE_TUNE_LAST_N > 0 else classifier_head.to(DEVICE)

    # ... 由于特征已经预计算，我们只训练分类头。端到端微调需要不同的数据加载和训练循环。
    # 为了保持当前结构，我们先用预计算的特征训练分类头
    print("[INFO] 当前脚本结构适用于预计算特征。将仅训练分类头。")
    print("[INFO] 要实现端到端微调，需要修改数据加载和训练循环。")

    # 训练循环只针对分类头
    optimizer = optim.AdamW(classifier_head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_metrics = run_epoch(classifier_head, train_loader, criterion, optimizer, device=DEVICE)
        va_loss, va_metrics = run_epoch(classifier_head, val_loader, criterion, None, device=DEVICE)
        scheduler.step(va_loss)
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_cnt = 0
            best_state_dict = classifier_head.state_dict()
        else:
            patience_cnt += 1
        print(f"[{epoch:03d}/{EPOCHS}] Train Loss={tr_loss:.4f} F1={tr_metrics['f1']:.3f} | Val Loss={va_loss:.4f} F1={va_metrics.get('f1', 0):.3f} | Patience={patience_cnt}/{PATIENCE}")
        if patience_cnt >= PATIENCE:
            print(f"早停触发于 Epoch {epoch}.")
            break

    # 10. 加载最佳模型并进行最终评估
    if best_state_dict:
        classifier_head.load_state_dict(best_state_dict)
    
    te_loss, te_metrics = run_epoch(classifier_head, test_loader, criterion, None, device=DEVICE)
    print("\n--- 最终评估 (Test Set) ---")
    print(f"Loss: {te_loss:.4f} | Accuracy: {te_metrics['accuracy']:.4f} | F1: {te_metrics['f1']:.4f}")

    # 11. 保存结果和模型
    with open(os.path.join(SAVE_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    res = pd.DataFrame([
        dict(split="train", loss=tr_loss, **tr_metrics),
        dict(split="val",   loss=va_loss, **va_metrics),
        dict(split="test",  loss=te_loss, **te_metrics),
    ])
    res = res.round(4)
    res.to_csv(os.path.join(SAVE_DIR, "final_results_summary.csv"), index=False)

    print("最终性能评估结果:")
    print(res)
    print(f"\n模型和结果已保存至: {Path(SAVE_DIR).resolve()}")

if __name__ == "__main__":
    main()