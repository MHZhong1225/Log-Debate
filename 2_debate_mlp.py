# generate_debate_features.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import argparse

from dataset_log_with_context_twotower import LogWithContextDataset
from context_agent import ContextAgent
from transformers import AutoTokenizer, AutoModelForCausalLM

class TwoMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        input_dim = input_dim
        self.output_dim = hidden_dim // 2 

        self.log_tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )
        self.context_tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )
        fused_dim = self.output_dim * 2
        self.classifier_head = nn.Linear(fused_dim, num_classes)

    def forward(self, log_vec, ctx_vec):
        log_features = self.log_tower(log_vec)
        ctx_features = self.context_tower(ctx_vec)
        fused_features = torch.cat([log_features, ctx_features], dim=-1)
        logits = self.classifier_head(fused_features)
        return logits

# --- 配置 ---
DATASET = "HDFS"  # BGL HDFS Tbird Hadoop 
MLP1_MODEL_PATH = f"model/{DATASET}/{DATASET}_best_mlp.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.85
CLASS_MAP = {0: "Normal", 1: "Anomaly"}
# if DATASET == 'Hadoop':
#     CLASS_MAP = {0: "Normal", 1: "Network Disconnection", 2: "Machine Down", 3: "Disk Full"}
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# --- LLM 初始化 ---
print(f"[LLM] Loading model: {LLM_MODEL_NAME}")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME, trust_remote_code=True, device_map="auto", dtype=torch.float16
).eval()
if llm_tokenizer.pad_token_id is None: llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

def call_llm_to_analyze(log_content: str) -> str:
    """专注于生成分析文本（证据），而不是做决策"""
    prompt = f"""
As a senior log analyst, your task is to analyze the following log entry and provide a brief, neutral analysis of its potential implications. Do not make a final judgment (e.g., "Normal" or "Anomaly"). Instead, describe the evidence within the log.

[Log Entry]:
{log_content}

[Your Analysis]:
Provide a concise, one-sentence analysis focusing on key processes, actions, or states mentioned in the log.
"""
    messages = [{"role": "system", "content": "You are an expert system log analysis assistant."}, {"role": "user", "content": prompt}]
    chat_text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer(chat_text, return_tensors="pt").to(llm_model.device)
    with torch.no_grad():
        # output_ids = llm_model.generate(**inputs, max_new_tokens=128, do_sample=False)
        output_ids = llm_model.generate(
            **inputs, 
            max_new_tokens=128,
            temperature=0.1,
            do_sample=True,
        )
    out_text = llm_tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return out_text or "Analysis could not be generated."

def build_dataset(csv_path, split_name):
    df = pd.read_csv(csv_path)
    # 将 "test" 模式映射为 "inference" 模式
    mode = "inference" if split_name == "test" else split_name
    ds = LogWithContextDataset(raw_logs=df["content"].tolist(), labels=df["label"].tolist(), mode=mode, ctx_agent=ContextAgent(window_size=100, hist_agg="max"))
    return ds

def generate_features_for_split(split_name: str):
    """为指定的数据集 split (train/val/test) 生成辩论特征"""
    input_csv = f"./datasets/{DATASET}/log_{split_name}.csv"
    if split_name == "train":
        input_csv = f"./datasets/{DATASET}/log_train_1p0.csv"
    output_csv = f"./datasets/{DATASET}/{split_name}_for_mlp2.csv"
    
    # 检查输出文件是否已存在
    if os.path.exists(output_csv):
        print(f"\n--- Skipping {split_name} split (output file already exists: {output_csv}) ---")
        return
    
    print(f"\n--- Processing {split_name} split ---")
    
    # 1. 加载数据集
    print(f"[1/4] Loading data from {input_csv}...")
    dataset = build_dataset(input_csv, split_name)

    # 2. 加载训练好的 MLP-1 模型
    print("[2/4] Loading trained MLP-1 model...")
    model = TwoMLP(input_dim=384, hidden_dim=256, num_classes=len(CLASS_MAP)).to(DEVICE)
    model.load_state_dict(torch.load(MLP1_MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 3. 遍历数据，生成辩论文本
    print("[3/4] Generating debate text features...")
    debate_texts = []
    raw_logs = pd.read_csv(input_csv)["content"].tolist()

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Generating for {split_name}"):
            (log_vec, ctx_vec), _ = dataset[i]
            log_vec = log_vec.unsqueeze(0).to(DEVICE)
            ctx_vec = ctx_vec.unsqueeze(0).to(DEVICE)

            logits = model(log_vec, ctx_vec)
            probs = F.softmax(logits, dim=1)
            confidence, _ = torch.max(probs, dim=1)
            
            if confidence.item() < CONFIDENCE_THRESHOLD:
                debate_text = call_llm_to_analyze(raw_logs[i])
            else:
                debate_text = "High confidence case."
            
            debate_texts.append(debate_text)

    # 4. 保存新的数据集
    print("[4/4] Saving new dataset...")
    df = pd.read_csv(input_csv)
    df['debate_text'] = debate_texts
    df.to_csv(output_csv, index=False)
    print(f"Successfully saved {len(df)} rows to {output_csv}")

if __name__ == "__main__":
    generate_features_for_split("train")
    generate_features_for_split("val")
    generate_features_for_split("test") # 同时为测试集也生成，以便最终评估
