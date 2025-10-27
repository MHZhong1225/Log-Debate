import torch
from torch.utils.data import Dataset
from context_agent import ContextAgent
from tqdm import tqdm
import os
import hashlib
import pandas as pd
import gc

CACHE_DIR = "./enc_cache"

class LogWithContextDataset(Dataset):
    def __init__(self, raw_logs, labels=None, mode='train', ctx_agent=None, batch_encode_size=64, use_context=True):
        assert mode in ['train', 'val', 'inference']
        self.logs = list(raw_logs)
        self.labels = None if labels is None else list(labels)
        self.mode = mode
        self.ctx_agent = ctx_agent or ContextAgent(hist_agg="max")
        self.device = self.ctx_agent.device
        self.use_context = use_context

        if not hasattr(self.ctx_agent, "encoder"):
            raise RuntimeError("ContextAgent needs to have an 'encoder' attribute.")

        self.tokens_list, self.log_vecs = self._batch_encode_logs(self.logs, batch_encode_size)
        
        self.ctx_vecs = self._precompute_ctx_vecs()

        if self.labels is not None:
            self.labels = torch.as_tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        log_vec = self.log_vecs[idx]
        ctx_vec = self.ctx_vecs[idx] if self.ctx_vecs is not None else torch.zeros_like(log_vec)
        
        x = (log_vec, ctx_vec)

        if self.labels is None:
            return x
        else:
            return x, self.labels[idx]

    def _reset_ctx_agent_window(self):
        if hasattr(self.ctx_agent, "reset_window"):
            self.ctx_agent.reset_window()

    @torch.no_grad()
    def _batch_encode_logs(self, texts, batch_encode_size):
        cfg_str = f"logs-{len(texts)}-{batch_encode_size}"
        cache_name = hashlib.md5(cfg_str.encode()).hexdigest()[:16] + ".pt"
        cache_path = os.path.join(CACHE_DIR, cache_name)

        if os.path.exists(cache_path):
            print(f"[Cache] Loading encoded logs from {cache_path}")
            ckpt = torch.load(cache_path, map_location=self.device)
            return ckpt["tokens_list"], ckpt["log_vecs"]

        print(f"[Cache] Encoding logs -> {cache_path}")
        tokens_list = [str(t).strip().split() for t in texts]
        encoder = self.ctx_agent.encoder
        vec_chunks = []
        for i in tqdm(range(0, len(texts), batch_encode_size), desc="Encoding logs"):
            batch_texts = texts[i:i + batch_encode_size]
            emb = encoder.encode(
                batch_texts, convert_to_tensor=True, normalize_embeddings=True
            ).to(self.device, dtype=torch.float32)
            vec_chunks.append(emb)
        log_vecs = torch.cat(vec_chunks, dim=0)
        
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save({"tokens_list": tokens_list, "log_vecs": log_vecs.cpu()}, cache_path)
        return tokens_list, log_vecs

    # @torch.no_grad()
    # def _precompute_ctx_vecs(self):
        if not self.use_context:
            return None

        win_size = getattr(self.ctx_agent, "window_size", "na")
        hist_agg = getattr(self.ctx_agent, "hist_agg", "na")
        cfg_str = f"ctx-{len(self.logs)}-{self.mode}-{win_size}-{hist_agg}"
        cache_name = hashlib.md5(cfg_str.encode()).hexdigest()[:16] + "_ctx.pt"
        cache_path = os.path.join(CACHE_DIR, cache_name)

        if os.path.exists(cache_path):
            print(f"[Cache] Loading context vectors from {cache_path}")
            return torch.load(cache_path, map_location=self.device)

        self._reset_ctx_agent_window()
        ctx_vecs_list = []
        print(f"[Context] Computing context vectors for {len(self.tokens_list)} logs...")
        for i, log_vec in enumerate(tqdm(self.log_vecs, desc="Computing context")):
            tokens = self.tokens_list[i]
            
            # 先获取上下文，再更新窗口
            ctx_vec = self.ctx_agent.get_ctx_vec(log_vec)
            ctx_vecs_list.append(ctx_vec)
            self.ctx_agent.update_window(tokens, log_vec)

        ctx_vecs = torch.stack(ctx_vecs_list, dim=0)
        
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(ctx_vecs.cpu(), cache_path)
        print(f"Saved context vectors to cache: {cache_path}")
        return ctx_vecs
        
    @torch.no_grad()
    def _precompute_ctx_vecs(self):
        if not self.use_context:
            return None

        win_size = getattr(self.ctx_agent, "window_size", "na")
        hist_agg = getattr(self.ctx_agent, "hist_agg", "na")
        cfg_str = f"ctx-{len(self.logs)}-{self.mode}-{win_size}-{hist_agg}"
        cache_name = hashlib.md5(cfg_str.encode()).hexdigest()[:16] + "_ctx.pt"
        cache_path = os.path.join(CACHE_DIR, cache_name)
        if os.path.exists(cache_path):
            print(f"[Cache] Loading context vectors from {cache_path}")
            return torch.load(cache_path, map_location=self.device)

        self._reset_ctx_agent_window()
        ctx_vecs_list = []
        print(f"[Context] Computing context vectors for {len(self.tokens_list)} logs...")
        
        for i, log_vec in enumerate(tqdm(self.log_vecs, desc="Computing context")):
            tokens = self.tokens_list[i]
            
            ctx_vec = self.ctx_agent.get_ctx_vec()
            ctx_vecs_list.append(ctx_vec)
            
            self.ctx_agent.update_window(tokens, log_vec)

        ctx_vecs = torch.stack(ctx_vecs_list, dim=0)
        
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(ctx_vecs.cpu(), cache_path)
        print(f"Saved context vectors to cache: {cache_path}")
        return ctx_vecs