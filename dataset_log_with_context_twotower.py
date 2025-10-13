import torch
from torch.utils.data import Dataset
from context_agent import ContextAgent
from tqdm import tqdm
from collections import Counter

import os, hashlib, torch
from tqdm import tqdm
import pandas as pd                      # 只用于判断 NaN，可删

CACHE_DIR = "./enc_cache"               # 缓存目录，可挪到 __init__ 里做成可配

class LogWithContextDataset(Dataset):
    def __init__(self, raw_logs, labels=None, mode='train', ctx_agent=None, batch_encode_size=64, use_context=True):
        """
        raw_logs: List[str] 日志文本（按时间顺序）
        labels:  List[int] 或 None
        mode:    'train' / 'val' / 'inference'
        ctx_agent: 外部传入的 ContextAgent 实例（推荐）；若不传则内部创建一个
        """
        assert mode in ['train', 'val', 'inference']
        self.logs = list(raw_logs)
        self.labels = None if labels is None else list(labels)
        self.mode = mode
        self.ctx_agent = ctx_agent or ContextAgent()
        self.device = self.ctx_agent.device
        self.use_context = use_context
        if not (hasattr(self.ctx_agent, "encode_log") or hasattr(self.ctx_agent, "encoder")):
            raise RuntimeError(
                "ContextAgent 需要提供 encode_log(text)->(tokens, emb_tensor) "
                "或暴露 encoder.encode(batch, convert_to_tensor=True)。"
            )

        # 1) 批量把所有日志编码成 tokens 与 log_vec（禁用梯度，避免显存抖动）
        self.tokens_list, self.log_vecs = self._batch_encode_logs(self.logs, batch_encode_size=batch_encode_size)

        # 2) 顺序预计算每条样本的 ctx_vec（在推进当前样本进窗口之前取上下文）
        self.ctx_vecs = self._precompute_ctx_vecs()

        # 3) 标签张量（若有）
        if self.labels is not None:
            y = torch.as_tensor(self.labels, dtype=torch.float32)  # 二分类用 BCEWithLogitsLoss 时 float 更方便
            if y.ndim == 1:
                y = y.unsqueeze(1)  # (N,1)
            self.labels = y
        if self.use_context:
            self.ctx_vecs = self._precompute_ctx_vecs()  # 以前的逻辑
        else:
            self.ctx_vecs = None  # 关掉上下文

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        log_vec = self.log_vecs[idx]      # (D,)
        ctx_vec = self.ctx_vecs[idx]      # (D,)

        # ✨ 核心修改：不再拼接，而是返回一个包含两个张量的元组
        # 外层的元组 ((log_vec, ctx_vec), label) 是 PyTorch DataLoader 的标准格式
        if self.labels is None:
            return log_vec, ctx_vec
        else:
            return (log_vec, ctx_vec), self.labels[idx]

    @torch.no_grad()
    def _batch_encode_logs(self, texts, batch_encode_size=64):
        """
        参数
        -------
        texts : List[str]
        batch_encode_size : int

        返回
        -------
        tokens_list : List[List[str]]
        log_vecs    : torch.FloatTensor  [N, D]  (已移动到 self.device)
        """
        # -------- 0) 生成 cache key --------
        cfg_str = f"{len(texts)}-{batch_encode_size}-ctx{self.use_context}"
        cache_name = hashlib.md5(cfg_str.encode()).hexdigest()[:16] + ".pt"
        cache_path = os.path.join(CACHE_DIR, cache_name)

        # -------- 1) 若缓存存在 → 直接加载 --------
        if os.path.exists(cache_path):
            print(f"[Cache] Loading encoded logs from {cache_path}")
            try:
                ckpt = torch.load(cache_path, map_location=self.device)
                print(f"[Cache] Successfully loaded {len(ckpt['tokens_list'])} logs")
                return ckpt["tokens_list"], ckpt["log_vecs"]
            except Exception as e:
                print(f"[Cache] Failed to load cache: {e}. Re-encoding...")
                # 删除损坏的缓存文件
                try:
                    os.remove(cache_path)
                except:
                    pass

        print(f"[Cache] Encoding logs → {cache_path}")

        # -------- 3) 分词 --------
        tokens_list = [t.strip().split() for t in texts]

        # -------- 4) 批量向量化 --------
        encoder = self.ctx_agent.encoder
        vec_chunks = []
        for i in tqdm(range(0, len(texts), batch_encode_size), desc="Encoding logs"):
            batch_texts = texts[i:i + batch_encode_size]
            emb = encoder.encode(batch_texts,
                                 convert_to_tensor=True,
                                 normalize_embeddings=True)
            vec_chunks.append(emb.to(self.device, dtype=torch.float32))

        log_vecs = torch.cat(vec_chunks, dim=0)   # [N, D]

        # -------- 5) 保存缓存 --------
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save({"tokens_list": tokens_list,
                    "log_vecs":     log_vecs.cpu()},  # 存 CPU 节省显存
                   cache_path)

        return tokens_list, log_vecs

    @torch.no_grad()
    def _precompute_ctx_vecs(self):
        """
        顺序预计算每条样本的上下文向量：
        - 在“推进当前样本到窗口”之前，先取它应该看到的上下文向量。
        - 然后再 update_window(tokens_i, log_vec_i)，供下一条使用。
        注意：这里假设 ContextAgent.get_ctx_vec() 是“基于当前窗口状态、与顺序相关”的有状态实现。
              如果你的 ContextAgent 是“无状态检索式”的（例如 FAISS 检索 + 均值），
              也可以把 get_ctx_vec(log_vec_i) 改为带当前向量作为查询的版本。
        """
        ctx_vecs = []

        # --- 关键修复 2：先把窗口清空，避免跨 split/跨阶段污染 ---
        if hasattr(self.ctx_agent, "reset_window"):
            self.ctx_agent.reset_window()
        else:
            # 兼容：常见属性名清空
            if hasattr(self.ctx_agent, "deque_vecs"):
                try:
                    self.ctx_agent.deque_vecs.clear()
                except Exception:
                    self.ctx_agent.deque_vecs = []
            if hasattr(self.ctx_agent, "deque_tokens"):
                try:
                    self.ctx_agent.deque_tokens.clear()
                except Exception:
                    self.ctx_agent.deque_tokens = []
            if hasattr(self.ctx_agent, "token_counter"):
                try:
                    self.ctx_agent.token_counter.clear()
                except Exception:
                    self.ctx_agent.token_counter = Counter()

        # 顺序推进
        print(f"[Context] Computing context vectors for {len(self.tokens_list)} logs...")
        for i, (tokens, log_vec) in enumerate(tqdm(zip(self.tokens_list, self.log_vecs), total=len(self.tokens_list), desc="Computing context")):
            # --- 关键修复 3：先取"当前可见的上下文" ---
            # 若你的 ContextAgent 是无状态检索(get_ctx_vec需要当前log向量作为查询)，改为：
            # ctx_vec = self.ctx_agent.get_ctx_vec(log_vec)
            if hasattr(self.ctx_agent, "get_ctx_vec"):
                try:
                    ctx_vec = self.ctx_agent.get_ctx_vec()  # 有状态滑窗：不带参数
                except TypeError:
                    # 兼容：如果你实现的是 get_ctx_vec(query_vec)
                    ctx_vec = self.ctx_agent.get_ctx_vec(log_vec)
            else:
                # 兜底：没有上下文就给 0 向量
                ctx_vec = torch.zeros_like(log_vec)

            ctx_vecs.append(ctx_vec.detach().clone().to(self.device, dtype=torch.float32))

            # --- 关键修复 4：再把"当前样本"推进窗口，供下一条使用 ---
            if hasattr(self.ctx_agent, "update_window"):
                self.ctx_agent.update_window(tokens, log_vec)
            # 否则无窗口可更新（无状态检索式），无需动作

        print(f"[Context] Successfully computed {len(ctx_vecs)} context vectors")
        return torch.stack(ctx_vecs, dim=0)  # [N, D]
