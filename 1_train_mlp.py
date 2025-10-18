import os

from model import CoherenceModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, average_precision_score
import argparse

# 确保以下两个文件与此脚本位于同一目录或正确的 Python 路径中
from dataset_log_with_context import LogWithContextDataset
from context_agent import ContextAgent

def parse_args():
    p = argparse.ArgumentParser(description="Train Hybrid Coherence Model (Classification + Coherence Loss)")
    p.add_argument("--dataset",      default="BGL", help="Dataset name: BGL, HDFS, Tbird, Hadoop")
    p.add_argument("--batch_size",   type=int,   default=32, help="Batch size for training")
    p.add_argument("--epochs",       type=int,   default=20, help="Number of training epochs")
    p.add_argument("--lr",           type=float, default=5e-4, help="Learning rate")
    
    # Coherence Loss Hyperparameters
    p.add_argument("--alpha",        type=float, default=1.0, help="Weight for alignment loss")
    p.add_argument("--beta",         type=float, default=0.1, help="Weight for variance loss")
    p.add_argument("--delta",        type=float, default=0.1, help="Weight for covariance loss")
    p.add_argument("--eta",          type=float, default=0.5, help="Weight for score margin loss")
    p.add_argument("--gamma",        type=float, default=1.0, help="Target stddev for variance loss")
    p.add_argument("--margin",       type=float, default=0.8, help="Score margin for normal samples")

    # 新增：混合损失的权重
    p.add_argument("--lambda_coeff", "-l_c", type=float, default=1, help="Weight for the coherence loss component")
    return p.parse_args()

args = parse_args()

# --- 全局常量 ---
DATASET     = args.dataset
BATCH_SIZE  = args.batch_size
EPOCHS      = args.epochs
LR          = args.lr
MODEL_PATH  = f"model/{DATASET}/{DATASET}_model.pt" # 新的模型路径
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CSV = f"./datasets/{DATASET}/log_train_1p0.csv"
VAL_CSV   = f"./datasets/{DATASET}/log_val_1p0.csv"
TEST_CSV  = f"./datasets/{DATASET}/log_test.csv"
if DATASET=='Tbird':
    VAL_CSV   = f"./datasets/{DATASET}/log_val_1p0.csv"
    TEST_CSV  = f"./datasets/{DATASET}/log_test_1p0.csv"
NORMAL_LABEL = 0
NUM_CLASSES = 2


# ================== 2. 损失函数定义 (保持不变) ==================
def alignment_loss(z_e, z_c): return (1 - F.cosine_similarity(z_e, z_c)).mean()
def variance_loss(z, gamma=1.0): return F.relu(gamma - torch.sqrt(z.var(dim=0) + 1e-4)).mean()
def covariance_loss(z):
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (z.size(0) - 1)
    off_diag_cov = cov.flatten()[:-1].view(z.size(1) - 1, z.size(1) + 1)[:, 1:].flatten()
    return off_diag_cov.pow(2).mean()
def score_margin_loss(score, labels, margin=0.8):
    mask = (labels == NORMAL_LABEL).float()
    loss = F.relu(margin - score) * mask
    return loss.sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0, device=score.device)

# ================== 3. 评估函数 (保持不变) ==================
def eval_model(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for (log_vec, ctx_vec), labels in loader:
            log_vec, ctx_vec = log_vec.to(device), ctx_vec.to(device)
            _, logits, _, _ = model(log_vec, ctx_vec)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu().numpy())
            
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = np.concatenate(all_labels).flatten()
    
    y_true_binary = (all_labels != NORMAL_LABEL).astype(int)
    y_pred = torch.argmax(all_logits, dim=1).numpy()
    anomaly_probs = F.softmax(all_logits, dim=1)[:, 1].numpy()

    f1 = f1_score(y_true_binary, y_pred)
    acc = accuracy_score(y_true_binary, y_pred)
    auroc = roc_auc_score(y_true_binary, anomaly_probs)
    auprc = average_precision_score(y_true_binary, anomaly_probs)
    
    return f1, acc, auroc, auprc, y_true_binary, y_pred

# ================== 主逻辑 ==================
def build_dataset(csv_path, split_name):
    df = pd.read_csv(csv_path)
    df['label'] = df['label'].astype(int)
    return LogWithContextDataset(
        raw_logs=df["content"].tolist(), labels=df["label"].tolist(),
        mode=split_name, ctx_agent=ContextAgent(window_size=100),
        batch_encode_size=64
    )

if __name__ == "__main__":
    print(f"Training on {DATASET} with device {DEVICE}")

    train_ds, val_ds, test_ds = [build_dataset(p, m) for p, m in zip([TRAIN_CSV, VAL_CSV, TEST_CSV], ["train", "val", "inference"])]
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    #  --- 计算加权损失 ---
    # print("Calculating weights for CrossEntropyLoss...")
    # # 1. 获取训练集的所有二分类标签 (0 for normal, 1 for anomaly)
    # y_train_binary = (train_ds.labels.squeeze() != NORMAL_LABEL).long()

    # # 2. 统计每个类别的数量
    # class_counts = Counter(y_train_binary.tolist())
    # print(f"Training class counts: {class_counts}")

    # # 3. 计算权重 (数量越少，权重越大)
    # weights = torch.tensor(
    #     [class_counts.get(i, 0) for i in range(NUM_CLASSES)],
    #     dtype=torch.float
    # )
    # weights = 1.0 / torch.clamp(weights, min=1.0)  # Use min=1 to avoid division by zero
    # weights = weights / weights.sum() * NUM_CLASSES    # 归一化到均值≈1
    # weights = weights.to(DEVICE)
    # print(f"Calculated weights: {weights.cpu().numpy()}")

    model = CoherenceModel(input_dim=384, proj_dim=128, hidden_dim=256, num_classes=2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # 使用计算出的权重初始化损失函数
    # classification_criterion = nn.CrossEntropyLoss(weight=weights)
    classification_criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for (log_vec, ctx_vec), labels in train_loader:
            log_vec, ctx_vec, labels = log_vec.to(DEVICE), ctx_vec.to(DEVICE), labels.squeeze().to(DEVICE)
            
            score, logits, z_e, z_c = model(log_vec, ctx_vec)
            
            l_align = alignment_loss(z_e, z_c)
            l_var = variance_loss(z_e, args.gamma) + variance_loss(z_c, args.gamma)
            # l_cov = covariance_loss(z_e) + covariance_loss(z_c)
            l_score = score_margin_loss(score, labels, args.margin)
            # loss_coherence = (args.alpha*l_align + args.beta*l_var + args.delta*l_cov + args.eta*l_score)
            loss_coherence = (args.alpha*l_align + args.beta*l_var + args.eta*l_score)

            y_true_binary = (labels != NORMAL_LABEL).long()
            loss_clf = classification_criterion(logits, y_true_binary)

            loss = loss_clf + args.lambda_coeff * loss_coherence

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        val_f1, val_acc, val_auroc, val_auprc, _, _ = eval_model(model, val_loader, DEVICE)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"[✓] Saved best model with F1: {best_f1:.4f}")

    print("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    test_f1, test_acc, test_auroc, test_auprc, y_true, y_pred = eval_model(model, test_loader, DEVICE)
    
    print(f"\n[Final Results for {DATASET}]")
    print(f"Test F1 (Anomaly):          {test_f1:.4f}\nTest Accuracy:    {test_acc:.4f}")
    print(f"Test AUROC:       {test_auroc:.4f}\nTest AUPRC:       {test_auprc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'], digits=4))