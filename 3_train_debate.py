# train_debate.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score

# 导入 MLP-1 的 Dataset 以复用编码
from dataset_log_with_context_twotower import LogWithContextDataset as MLP1Dataset
from context_agent import ContextAgent

# --- 1. 定义新的数据集类 for MLP-2 ---
class DebateDataset(Dataset):
    def __init__(self, csv_path, mlp1_dataset, device='cuda'):
        self.df = pd.read_csv(csv_path)
        self.mlp1_dataset = mlp1_dataset # 复用已编码的 log_vec 和 ctx_vec
        self.device = device
        
        # 检查是否已有缓存的编码文件
        cache_path = csv_path.replace('.csv', '_debate_vecs.pt')
        if os.path.exists(cache_path):
            print(f"Loading cached debate vectors from {cache_path}...")
            try:
                self.debate_vecs = torch.load(cache_path, map_location='cpu')
                print(f"Successfully loaded {len(self.debate_vecs)} debate vectors")
            except Exception as e:
                print(f"Failed to load cache: {e}. Re-encoding...")
                self._encode_debate_texts(csv_path, device)
        else:
            self._encode_debate_texts(csv_path, device)
    
    def _encode_debate_texts(self, csv_path, device):
        """编码辩论文本并保存缓存"""
        print(f"Encoding debate texts for {csv_path}...")
        encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        
        # 使用GPU编码，但立即移动到CPU避免多进程问题
        self.debate_vecs = encoder.encode(
            self.df['debate_text'].astype(str).tolist(), 
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=64  # 增加批处理大小提高效率
        ).cpu()  # 编码完成后立即移动到CPU
        
        # 保存缓存
        cache_path = csv_path.replace('.csv', '_debate_vecs.pt')
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(self.debate_vecs, cache_path)
        print(f"Cached debate vectors saved to {cache_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        (log_vec, ctx_vec), label = self.mlp1_dataset[idx]
        debate_vec = self.debate_vecs[idx]  # 从CPU获取
        
        # 确保标签是标量
        if isinstance(label, torch.Tensor):
            label = label.item() if label.numel() == 1 else label.squeeze().item()
        else:
            label = int(label)
            
        return (log_vec, ctx_vec, debate_vec), torch.tensor(label, dtype=torch.long)

# --- 2. 定义新的三塔模型 MLP-2 ---
class ThreeTowerMLP(nn.Module):
    def __init__(self, log_ctx_dim, debate_dim, hidden_dim, num_classes):
        super().__init__()
        # 3个独立的塔
        self.log_tower = nn.Linear(log_ctx_dim, hidden_dim // 2)
        self.ctx_tower = nn.Linear(log_ctx_dim, hidden_dim // 2)
        self.debate_tower = nn.Linear(debate_dim, hidden_dim // 2)
        
        fused_dim = (hidden_dim // 2) * 3
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, num_classes)
        )

    def forward(self, log_vec, ctx_vec, debate_vec):
        log_feat = self.log_tower(log_vec)
        ctx_feat = self.ctx_tower(ctx_vec)
        debate_feat = self.debate_tower(debate_vec)
        
        fused = torch.cat([log_feat, ctx_feat, debate_feat], dim=-1)
        logits = self.classifier_head(fused)
        return logits

# --- 3. 训练和评估主逻辑 ---
def main():
    # --- 配置 ---
    DATASET = "Tbird" # BGL HDFS Tbird Hadoop
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-4
    MODEL_PATH = f"model/{DATASET}/{DATASET}_best_mlp.pt"

    # --- 数据加载 ---
    print("\n--- Loading Data for MLP-2 ---")
    
    # 先加载 MLP-1 的数据集以复用编码
    print("Loading MLP-1 training dataset...")
    train_df = pd.read_csv(f"./datasets/{DATASET}/log_train_1p0.csv")
    mlp1_train_ds = MLP1Dataset(raw_logs=train_df["content"], labels=train_df["label"], mode='train', ctx_agent=ContextAgent())
    
    print("Loading MLP-1 validation dataset...")
    val_df = pd.read_csv(f"./datasets/{DATASET}/log_val.csv")
    mlp1_val_ds = MLP1Dataset(raw_logs=val_df["content"], labels=val_df["label"], mode='val', ctx_agent=ContextAgent())
    
    # 再用增强的 CSV 和 MLP-1 数据集创建 MLP-2 数据集
    train_ds = DebateDataset(f'./datasets/{DATASET}/train_for_mlp2.csv', mlp1_train_ds, device=DEVICE)
    val_ds = DebateDataset(f'./datasets/{DATASET}/val_for_mlp2.csv', mlp1_val_ds, device=DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 模型和优化器 ---
    model = ThreeTowerMLP(log_ctx_dim=384, debate_dim=384, hidden_dim=512, num_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # --- 训练循环 ---
    print("\n--- Starting Training for MLP-2 ---")
    best_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for (log, ctx, debate), labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            logits = model(log.to(DEVICE), ctx.to(DEVICE), debate.to(DEVICE))
            loss = criterion(logits, labels.to(DEVICE))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # --- 验证 ---
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for (log, ctx, debate), labels in val_loader:
                logits = model(log.to(DEVICE), ctx.to(DEVICE), debate.to(DEVICE))
                y_pred.extend(logits.argmax(1).cpu().tolist())
                y_true.extend(labels.cpu().tolist())
        
        val_f1 = f1_score(y_true, y_pred, average='macro')
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"[✓] Saved best model with F1: {best_f1:.4f}")

    # --- 最终测试 ---
    print("\n--- Final Evaluation on Test Set ---")
    mlp1_test_ds = MLP1Dataset(raw_logs=pd.read_csv(f"./datasets/{DATASET}/log_test.csv")["content"], labels=pd.read_csv(f"./datasets/{DATASET}/log_test.csv")["label"], mode='inference', ctx_agent=ContextAgent())
    test_ds = DebateDataset(f'./datasets/{DATASET}/test_for_mlp2.csv', mlp1_test_ds, device=DEVICE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for (log, ctx, debate), labels in test_loader:
            logits = model(log.to(DEVICE), ctx.to(DEVICE), debate.to(DEVICE))
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(labels.cpu().tolist())
    
    print("Final Test Report for MLP-2:")
    print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    main()
