import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class ContextAgent(nn.Module):
    def __init__(self,
                 window_size=100,
                 hidden_dim=384,
                 hist_agg: str = "max",
                 device='cpu'):

        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.device = device

        # 滑动窗口缓存
        self.deque_vecs = collections.deque(maxlen=window_size)
        self.deque_tokens = collections.deque(maxlen=window_size)
        self.token_counter = collections.Counter()

        self.hist_agg = hist_agg.lower()

        # ✨ 新增：为注意力机制定义 Q, K, V 投影层
        if self.hist_agg == "attention":
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 最终的融合层 (保持不变)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        # 编码器
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")


    def reset_window(self):
        """新增一个显式清空窗口的方法，更规范"""
        self.deque_vecs.clear()
        self.deque_tokens.clear()
        self.token_counter.clear()

    def update_window(self, tokens: list, log_vec: torch.Tensor):
        # 确保传入的是 tensor
        if isinstance(log_vec, np.ndarray):
            log_vec = torch.tensor(log_vec, device=self.device, dtype=torch.float32)
        
        self.deque_vecs.append(log_vec)
        self.deque_tokens.append(tokens)
        self.token_counter.update(tokens)


    def get_ctx_vec(self, current_log_vec: torch.Tensor) -> torch.Tensor:
        """
        ✨ 修改：方法签名改变，需要传入 current_log_vec 作为 Query
        生成上下文向量。
        """
        # 边界情况：如果窗口为空，没有历史信息
        if len(self.deque_vecs) == 0:
            # 使用一个零向量作为历史上下文
            hist_vec = torch.zeros_like(current_log_vec)
        else:
            # 将deque中的历史向量堆叠成一个张量
            hist_stack = torch.stack(list(self.deque_vecs), dim=0)  # (W, D)

            if self.hist_agg == "max":
                hist_vec, _ = torch.max(hist_stack, dim=0)
            elif self.hist_agg == "mean":
                hist_vec = hist_stack.mean(dim=0)
            elif self.hist_agg == "attention":
                # --- 注意力机制核心逻辑 ---
                # 1. Query, Key, Value 投影
                # Query 来自当前日志, Key/Value 来自历史
                q = self.q_proj(current_log_vec.unsqueeze(0)) # (1, D)
                k = self.k_proj(hist_stack)                   # (W, D)
                v = self.v_proj(hist_stack)                   # (W, D)

                # 2. 计算注意力分数 (scaled dot-product)
                d_k = q.size(-1)
                attn_scores = torch.matmul(q, k.transpose(0, 1)) / (d_k ** 0.5) # (1, W)

                # 3. 计算注意力权重
                attn_weights = F.softmax(attn_scores, dim=-1) # (1, W)

                # 4. 加权求和得到历史上下文向量
                hist_vec = torch.matmul(attn_weights, v).squeeze(0) # (1, D) -> (D,)
            else:
                raise ValueError(f"Unsupported hist_agg: {self.hist_agg}")
        
        # 拼接 + 线性融合 (current_log_vec 替换了原来的 current_vec)
        ctx_vec = torch.cat([current_log_vec, hist_vec], dim=-1) # (2D,)
        ctx_vec = self.proj(ctx_vec)                            # (D,)
        return ctx_vec


    def encode_log(self, raw_log: str):
        """将原始日志文本转换为向量和 tokens"""
        log_vec = self.encoder.encode(raw_log, normalize_embeddings=True)
        tokens = raw_log.strip().split()
        return tokens, log_vec