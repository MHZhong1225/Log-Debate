import torch
from torch.utils.data import Dataset
from context_agent import ContextAgent
from tqdm import tqdm

import os, hashlib, torch
from tqdm import tqdm
import pandas as pd
import gc

CACHE_DIR = "./enc_cache"

class LogWithContextDatasetMemoryOptimized(Dataset):
    def __init__(self, raw_logs, labels=None, mode='train', ctx_agent=None, batch_encode_size=16, use_context=True):
        """
        内存优化版本：不预计算所有context vectors，而是在__getitem__时动态计算
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

        # 只预计算log vectors，不预计算context vectors
        self.tokens_list, self.log_vecs = self._batch_encode_logs(self.logs, batch_encode_size=batch_encode_size)

        # 预计算一次 ctx_vecs，并缓存到磁盘，避免在 __getitem__ 中重复构建上下文导致的极慢速度
        self.ctx_vecs = self._precompute_ctx_vecs()

        # 标签张量（若有）
        if self.labels is not None:
            self.labels = torch.tensor(self.labels, dtype=torch.long, device=self.device)

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        log_vec = self.log_vecs[idx]
        tokens = self.tokens_list[idx]
        
        if self.use_context:
            if self.ctx_vecs is not None:
                ctx_vec = self.ctx_vecs[idx]
            else:
                # 兼容：如果没能成功预计算，就退回到旧的按需计算逻辑
                ctx_vec = self._get_context_vector_at_index(idx)
            x = (log_vec, ctx_vec)
        else:
            x = log_vec

        if self.labels is None:
            return x
        else:
            return x, self.labels[idx]

    def _reset_ctx_agent_window(self):
        """清空 ContextAgent 的窗口缓存。"""
        if hasattr(self.ctx_agent, "reset_window"):
            self.ctx_agent.reset_window()
        else:
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
                    counter_cls = type(self.ctx_agent.token_counter)
                    try:
                        self.ctx_agent.token_counter = counter_cls()
                    except Exception:
                        self.ctx_agent.token_counter = {}

    def _get_context_vector_at_index(self, idx):
        """
        动态计算指定索引处的context vector
        这里需要重新构建到该索引为止的context
        """
        # 重置context agent
        self._reset_ctx_agent_window()
        
        # 逐步构建到idx为止的context
        for i in range(idx):
            tokens_i = self.tokens_list[i]
            log_vec_i = self.log_vecs[i]
            self.ctx_agent.update_window(tokens_i, log_vec_i)
        
        # 获取当前索引的context vector
        if hasattr(self.ctx_agent, "get_ctx_vec"):
            try:
                ctx_vec = self.ctx_agent.get_ctx_vec()
            except TypeError:
                ctx_vec = self.ctx_agent.get_ctx_vec(self.log_vecs[idx])
        else:
            ctx_vec = torch.zeros_like(self.log_vecs[idx])
        
        return ctx_vec.detach().clone().to(self.device, dtype=torch.float32)

    @torch.no_grad()
    def _batch_encode_logs(self, texts, batch_encode_size=16):
        """
        批量编码日志，使用更小的batch size
        """
        cfg_str = f"{len(texts)}-{batch_encode_size}-ctx{self.use_context}"
        cache_name = hashlib.md5(cfg_str.encode()).hexdigest()[:16] + ".pt"
        cache_path = os.path.join(CACHE_DIR, cache_name)

        if os.path.exists(cache_path):
            print(f"[Cache] Loading encoded logs from {cache_path}")
            ckpt = torch.load(cache_path, map_location=self.device)
            return ckpt["tokens_list"], ckpt["log_vecs"]

        print(f"[Cache] Encoding logs → {cache_path}")

        tokens_list = [t.strip().split() for t in texts]

        encoder = self.ctx_agent.encoder
        vec_chunks = []
        
        # 使用更小的batch size并定期清理内存
        for i in tqdm(range(0, len(texts), batch_encode_size), desc="Encoding logs"):
            batch_texts = texts[i:i + batch_encode_size]
            emb = encoder.encode(batch_texts,
                                 convert_to_tensor=True,
                                 normalize_embeddings=True)
            vec_chunks.append(emb.to(self.device, dtype=torch.float32))
            
            # 每处理一定数量的batch后清理内存
            if i % (batch_encode_size * 10) == 0:
                gc.collect()

        log_vecs = torch.cat(vec_chunks, dim=0)

        # 保存缓存
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save({"tokens_list": tokens_list,
                    "log_vecs": log_vecs.cpu()},
                   cache_path)

        return tokens_list, log_vecs

    @torch.no_grad()
    def _precompute_ctx_vecs(self):
        """顺序预计算每条样本的上下文向量，并进行磁盘缓存。"""
        if not self.use_context:
            return None

        win_size = getattr(self.ctx_agent, "window_size", "na")
        hist_agg = getattr(self.ctx_agent, "hist_agg", "na")
        cfg_str = f"ctx-{len(self.logs)}-{self.mode}-{win_size}-{hist_agg}"
        cache_name = hashlib.md5(cfg_str.encode()).hexdigest()[:16] + "_ctx.pt"
        cache_path = os.path.join(CACHE_DIR, cache_name)

        if os.path.exists(cache_path):
            try:
                print(f"[Cache] Loading context vectors from {cache_path}")
                ctx_vecs = torch.load(cache_path, map_location=self.device)
                return ctx_vecs.to(self.device, dtype=torch.float32)
            except Exception as e:
                print(f"[Cache] Failed to load context cache: {e}. Recomputing...")
                try:
                    os.remove(cache_path)
                except Exception:
                    pass

        self._reset_ctx_agent_window()

        ctx_vecs = []
        print(f"[Context] Computing context vectors for {len(self.tokens_list)} logs...")
        for tokens, log_vec in tqdm(zip(self.tokens_list, self.log_vecs),
                                    total=len(self.tokens_list),
                                    desc="Computing context"):
            if hasattr(self.ctx_agent, "get_ctx_vec"):
                try:
                    ctx_vec = self.ctx_agent.get_ctx_vec()
                except TypeError:
                    ctx_vec = self.ctx_agent.get_ctx_vec(log_vec)
            else:
                ctx_vec = torch.zeros_like(log_vec)

            ctx_vecs.append(ctx_vec.detach().clone().to(self.device, dtype=torch.float32))

            if hasattr(self.ctx_agent, "update_window"):
                self.ctx_agent.update_window(tokens, log_vec)

        ctx_vecs = torch.stack(ctx_vecs, dim=0)

        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(ctx_vecs.cpu(), cache_path)

        return ctx_vecs.to(self.device, dtype=torch.float32)
