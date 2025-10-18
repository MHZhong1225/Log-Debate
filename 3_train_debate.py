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
from dataset_log_with_context import LogWithContextDataset as MLP1Dataset
from context_agent import ContextAgent
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Train Hybrid Coherence Model (Classification + Coherence Loss)")
    p.add_argument("--dataset",      default="Tbird", help="Dataset name: BGL, HDFS, Tbird, Hadoop")
    p.add_argument("--batch_size",   type=int,   default=128, help="Batch size for training")
    p.add_argument("--epochs",       type=int,   default=19, help="Number of training epochs")
    p.add_argument("--lr",           type=float, default=5e-4, help="Learning rate")
     
    return p.parse_args()
# In 3_train_debate.py -> DebateDataset class


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
    

    # def _encode_debate_texts(self, csv_path, device):
    #     print(f"Encoding debate texts for {csv_path} with special handling...")
    #     encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        
    #     all_texts = self.df['debate_text'].astype(str).tolist()
    #     placeholder = "High confidence case."
        
    #     indices_to_encode = [i for i, text in enumerate(all_texts) if text != placeholder]
    #     texts_to_encode = [all_texts[i] for i in indices_to_encode]

    #     print(f"Found {len(texts_to_encode)} out of {len(all_texts)} logs requiring LLM debate encoding.")

    #     # 初始化一个全零的张量来存放所有向量
    #     embedding_dim = 384 # all-MiniLM-L6-v2 的维度
    #     all_vecs = torch.ones(len(all_texts), embedding_dim, dtype=torch.float32)

    #     # 仅对需要编码的文本进行编码
    #     if texts_to_encode:
    #         real_embeddings = encoder.encode(
    #             texts_to_encode,
    #             convert_to_tensor=True,
    #             show_progress_bar=True,
    #             normalize_embeddings=True,
    #             batch_size=64
    #         ).cpu()
            
    #         # 将编码好的向量放回它们在全零张量中的正确位置
    #         all_vecs[torch.tensor(indices_to_encode)] = real_embeddings

    #     self.debate_vecs = all_vecs
        
    #     # 保存缓存
    #     cache_path = csv_path.replace('.csv', '_debate_vecs.pt') # 使用新名字避免混淆
    #     os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    #     torch.save(self.debate_vecs, cache_path)
    #     print(f"Cached debate vectors saved to {cache_path}")
    
    def _encode_debate_texts(self, csv_path, device):

        cache_path = csv_path.replace('.csv', '_debate_vecs_v4_gated_zero.pt')
        if os.path.exists(cache_path):
            print(f"Loading cached debate vectors from {cache_path}...")
            self.debate_vecs = torch.load(cache_path, map_location='cpu')
            return

        print(f"Encoding debate texts for {csv_path} with ZERO vector fallback...")
        encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        
        all_texts = self.df['debate_text'].astype(str).tolist()
        placeholder = "High confidence case."
        
        indices_to_encode = [i for i, text in enumerate(all_texts) if text != placeholder]
        texts_to_encode = [all_texts[i] for i in indices_to_encode]

        print(f"Found {len(texts_to_encode)} out of {len(all_texts)} logs requiring LLM debate encoding.")

        embedding_dim = 384
        all_vecs = torch.zeros(len(all_texts), embedding_dim, dtype=torch.float32)

        if texts_to_encode:
            real_embeddings = encoder.encode(
                texts_to_encode,
                convert_to_tensor=True,
                show_progress_bar=True,
                normalize_embeddings=True,
                batch_size=64
            ).cpu()
            all_vecs[torch.tensor(indices_to_encode)] = real_embeddings

        self.debate_vecs = all_vecs
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(self.debate_vecs, cache_path)
        print(f"Cached debate vectors (ZERO fallback) saved to {cache_path}")

    
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

# --- 2. 定义新的模型 MLP-2 ---
# class ThreeMLP(nn.Module):
#     def __init__(self, log_ctx_dim, debate_dim, hidden_dim, num_classes):
#         super().__init__()

#         self.log = nn.Sequential(
#             nn.Linear(log_ctx_dim, hidden_dim),
#             nn.ReLU(),
#             # nn.BatchNorm1d(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim // 2)
#         )
#         self.ctx = nn.Sequential(
#             nn.Linear(log_ctx_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim // 2)
#         )

#         # self.log = nn.Linear(log_ctx_dim, hidden_dim // 2)
#         # self.ctx = nn.Linear(log_ctx_dim, hidden_dim // 2)
#         self.debate = nn.Linear(debate_dim, hidden_dim // 2)
        
#         fused_dim = (hidden_dim // 2) * 3
#         self.classifier_head = nn.Sequential(
#             nn.LayerNorm(fused_dim),
#             nn.ReLU(),
#             nn.Linear(fused_dim, num_classes)
#         )

#     def forward(self, log_vec, ctx_vec, debate_vec):
#         log_feat = self.log(log_vec)
#         ctx_feat = self.ctx(ctx_vec)
#         debate_feat = self.debate(debate_vec)
        
#         fused = torch.cat([log_feat, ctx_feat, debate_feat], dim=-1)
#         logits = self.classifier_head(fused)
#         return logits


# 在 3_train_debate.py 中替换此模型 (v5: Gated Addition)
class ThreeMLP(nn.Module):
    def __init__(self, log_ctx_dim, debate_dim, hidden_dim, num_classes):
        super().__init__()

        # 将 log 和 debate 分支的输出维度统一
        self.log_proj_dim = hidden_dim // 2
        self.ctx_proj_dim = hidden_dim // 2
        
        # log 分支
        self.log = nn.Sequential(
            nn.Linear(log_ctx_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.log_proj_dim) # 输出 [B, 128]
        )
        
        # context 分支
        self.ctx = nn.Sequential(
            nn.Linear(log_ctx_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.ctx_proj_dim) # 输出 [B, 128]
        )

        # debate 分支 (与 log 分支输出维度相同)
        self.debate = nn.Sequential(
            nn.Linear(debate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.log_proj_dim) # 输出 [B, 128]
        )
        
        # 新的融合维度：(log+debate) + ctx
        fused_dim = self.log_proj_dim + self.ctx_proj_dim # 128 + 128 = 256
        
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, num_classes)
        )

    def forward(self, log_vec, ctx_vec, debate_vec):
        # 1. 独立计算三个特征
        log_feat = self.log(log_vec)       # [B, 128]
        ctx_feat = self.ctx(ctx_vec)       # [B, 128]
        debate_feat_raw = self.debate(debate_vec) # [B, 128]

        # 2. 创建门 (gate)
        #    torch.any(debate_vec != 0, dim=1, keepdim=True) -> [B, 1] (True/False)
        gate = torch.any(debate_vec != 0, dim=1, keepdim=True).float() # [B, 1] (1.0/0.0)

        # 3. 门控加法 (Gated Addition)
        #    用 debate_feat 来“修正” log_feat
        #    如果 gate=0, refined_log = log_feat
        #    如果 gate=1, refined_log = log_feat + debate_feat_raw
        refined_log_feat = log_feat + (debate_feat_raw * gate)

        # 4. 拼接
        fused = torch.cat([refined_log_feat, ctx_feat], dim=-1)
        
        # 5. 分类
        logits = self.classifier_head(fused)
        return logits


# --- 3. 训练和评估主逻辑 ---
def main():
    args = parse_args()
    # --- 全局常量 ---
    DATASET     = args.dataset
    BATCH_SIZE  = args.batch_size
    EPOCHS      = args.epochs
    LR          = args.lr

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = f"model/{DATASET}/{DATASET}_best_mlp.pt"

    # --- 数据加载 ---
    print("\n--- Loading Data for MLP-2 ---")
    
    # 先加载 MLP-1 的数据集以复用编码
    print("Loading MLP-1 training dataset...")
    train_df = pd.read_csv(f"./datasets/{DATASET}/log_train_1p0.csv")
    mlp1_train_ds = MLP1Dataset(raw_logs=train_df["content"], labels=train_df["label"], mode='train', ctx_agent=ContextAgent())
    
    print("Loading MLP-1 validation dataset...")
    val_df = pd.read_csv(f"./datasets/{DATASET}/log_val_1p0.csv")
    mlp1_val_ds = MLP1Dataset(raw_logs=val_df["content"], labels=val_df["label"], mode='val', ctx_agent=ContextAgent())
    
    # 再用增强的 CSV 和 MLP-1 数据集创建 MLP-2 数据集
    train_ds = DebateDataset(f'./datasets/{DATASET}/train_debate.csv', mlp1_train_ds, device=DEVICE)
    val_ds = DebateDataset(f'./datasets/{DATASET}/val_debate.csv', mlp1_val_ds, device=DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 模型和优化器 ---
    model = ThreeMLP(log_ctx_dim=384, debate_dim=384, hidden_dim=256, num_classes=2).to(DEVICE)
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
    df = pd.read_csv(f"./datasets/{DATASET}/log_test.csv")
    if DATASET == 'Tbird':
        df = pd.read_csv(f"./datasets/{DATASET}/log_test_1p0.csv")
    mlp1_test_ds = MLP1Dataset(raw_logs=df["content"], labels=df["label"], mode='inference', ctx_agent=ContextAgent())
    test_ds = DebateDataset(f'./datasets/{DATASET}/test_debate.csv', mlp1_test_ds, device=DEVICE)
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
