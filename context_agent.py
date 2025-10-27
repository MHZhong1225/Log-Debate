import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class ContextAgent(nn.Module):
    def __init__(self,
                 window_size=120,
                 hidden_dim=384,
                 hist_agg: str = "max",
                 device='cpu'):

        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.device = device

        self.deque_vecs = collections.deque(maxlen=window_size)
        self.deque_tokens = collections.deque(maxlen=window_size)
        self.token_counter = collections.Counter()

        self.hist_agg = hist_agg.lower()



        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")


    def reset_window(self):
        self.deque_vecs.clear()
        self.deque_tokens.clear()
        self.token_counter.clear()

    def update_window(self, tokens: list, log_vec: torch.Tensor):
        if isinstance(log_vec, np.ndarray):
            log_vec = torch.tensor(log_vec, device=self.device, dtype=torch.float32)
        
        self.deque_vecs.append(log_vec)
        self.deque_tokens.append(tokens)
        self.token_counter.update(tokens)


    # def get_ctx_vec(self, current_log_vec: torch.Tensor) -> torch.Tensor:
    #     if len(self.deque_vecs) == 0:
    #         hist_vec = torch.zeros_like(current_log_vec)
    #     else:
    #         hist_stack = torch.stack(list(self.deque_vecs), dim=0)  # (W, D)

    #         if self.hist_agg == "max":
    #             hist_vec, _ = torch.max(hist_stack, dim=0)
    #         elif self.hist_agg == "mean":
    #             hist_vec = hist_stack.mean(dim=0)
    #         elif self.hist_agg == "attention":
    #             q = self.q_proj(current_log_vec.unsqueeze(0)) # (1, D)
    #             k = self.k_proj(hist_stack)                   # (W, D)
    #             v = self.v_proj(hist_stack)                   # (W, D)

    #             d_k = q.size(-1)
    #             attn_scores = torch.matmul(q, k.transpose(0, 1)) / (d_k ** 0.5) # (1, W)

    #             attn_weights = F.softmax(attn_scores, dim=-1) # (1, W)

    #             hist_vec = torch.matmul(attn_weights, v).squeeze(0) # (1, D) -> (D,)
    #         else:
    #             raise ValueError(f"Unsupported hist_agg: {self.hist_agg}")
        
    #     ctx_vec = torch.cat([current_log_vec, hist_vec], dim=-1) # (2D,)
    #     ctx_vec = self.proj(ctx_vec)                            # (D,)
    #     return ctx_vec

    def get_ctx_vec(self) -> torch.Tensor:
            if len(self.deque_vecs) == 0:
                hist_vec = torch.zeros(self.hidden_dim, device=self.device, dtype=torch.float32)
            else:
                hist_stack = torch.stack(list(self.deque_vecs), dim=0)  # (W, D)

                if self.hist_agg == "max":
                    hist_vec, _ = torch.max(hist_stack, dim=0)
                elif self.hist_agg == "mean":
                    hist_vec = hist_stack.mean(dim=0)
                else:
                    raise ValueError(f"Unsupported hist_agg: {self.hist_agg}")                     
            
            return hist_vec

    def encode_log(self, raw_log: str):
        log_vec = self.encoder.encode(raw_log, normalize_embeddings=True)
        tokens = raw_log.strip().split()
        return tokens, log_vec
        