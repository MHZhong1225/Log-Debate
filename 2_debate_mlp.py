# 2_debate_mlp.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import argparse

# 确保这些自定义模块可以被导入
from dataset_log_with_context import LogWithContextDataset
from context_agent import ContextAgent
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import CoherenceModel

import random
from torch.utils.data import DataLoader

def parse_args():
    p = argparse.ArgumentParser(description="Generate Debate Features using a trained MLP-1 and an LLM")
    p.add_argument("--dataset", default="BGL", help="Dataset name: BGL, HDFS, Tbird, Hadoop")
    p.add_argument("--threshold", type=float, default=0.95, help="Confidence threshold to trigger LLM debate")
    p.add_argument("--llm_context_window", type=int, default=5, help="Number of preceding logs for LLM context")
    return p.parse_args()

# --- LLM 调用函数 (核心修改) ---
def call_llm_for_debate(target_log: str, context_logs: list[str], tokenizer, model) -> str:
    context_str = "\n".join(f"- {log}" for log in context_logs) if context_logs else "No preceding context available."

    prompt = f"""
As an expert Site Reliability Engineer (SRE), your task is to conduct a structured, evidence-based debate on whether the following "Target Log" is anomalous. Your reasoning must be based *only* on the "Surrounding Context" provided.

### Instructions:
1.  **Analyze the sequence and semantics**: Compare the Target Log with the events that came before it.
2.  **Generate opposing arguments**: You must provide evidence for *both* sides of the debate.
3.  **Remain neutral**: Do not give a final verdict. Your role is to present the evidence for a human operator to decide.

### Inputs:
[Surrounding Context]:
{context_str}

[Target Log to Analyze]:
{target_log}

### Structured Debate Output:
**Evidence for Anomaly:**
(Based on the context, provide 1-2 concise bullet points arguing why the Target Log could be anomalous. Focus on broken sequences, unexpected states, contradictions, or deviations.)
-

**Evidence for Normality:**
(Based on the context, provide 1-2 concise bullet points arguing why the Target Log could be normal. Focus on routine operations, expected follow-up actions, or lack of explicit errors.)
-

**Neutral Summary:**
(Provide a single sentence that summarizes the core conflict in the evidence without taking a side.)
"""
    messages = [{"role": "system", "content": "You are an expert system log analysis assistant."}, {"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True)
    out_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return out_text or "Analysis could not be generated."

def build_dataset(csv_path, split_name):
    df = pd.read_csv(csv_path)
    mode = "inference" if split_name == "test" else split_name
    ds = LogWithContextDataset(raw_logs=df["content"].tolist(), labels=df["label"].tolist(), mode=mode, ctx_agent=ContextAgent(window_size=100))
    return ds

# def generate_features_for_split(args, llm_tokenizer, llm_model):
#     """为指定的数据集 split (train/val/test) 生成辩论特征"""
#     split_name = args.split
#     DATASET = args.dataset
    
#     # 统一文件路径逻辑
#     # base_path = f"./datasets/{DATASET}/log_{split_name}_1p0"
#     input_csv = f"./datasets/{DATASET}/log_{split_name}_1p0.csv"
#     # input_csv = f"./datasets/{DATASET}/log_val_1p0.csv"
#     # elif DATASET == "Tbird" and split_name != "train":
#     # if DATASET == "Tbird":
#     #     input_csv = f"{base_path}_1p0.csv"
#     # else:
#     #     input_csv = f"{base_path}.csv"

#     if split_name == 'test':
#         input_csv = f"./datasets/{DATASET}/log_{split_name}.csv"


#     output_csv = f"./datasets/{DATASET}/{split_name}_debate.csv"
    
#     if os.path.exists(output_csv):
#         print(f"\n--- Skipping {split_name} split (output file already exists: {output_csv}) ---")
#         return
    
#     print(f"\n--- Processing {split_name} split ---")
    
#     print(f"[1/4] Loading data from {input_csv}...")
#     dataset = build_dataset(input_csv, split_name)
#     raw_logs = pd.read_csv(input_csv)["content"].tolist()

#     print("[2/4] Loading trained MLP-1 model...")
#     # 动态确定类别数
#     num_classes = len(pd.read_csv(input_csv)['label'].unique())
#     # model = CoherenceModel(input_dim=384, hidden_dim=256, num_classes=num_classes).to(DEVICE)
#     model = CoherenceModel(input_dim=384, proj_dim=128, hidden_dim=256, num_classes=2).to(DEVICE)

#     model.load_state_dict(torch.load(args.mlp1_model_path, map_location=DEVICE))
#     model.eval()

#     print("[3/4] Generating debate text features...")
#     debate_texts = []
#     llm_call_count = 0
#     with torch.no_grad():
#         for i in tqdm(range(len(dataset)), desc=f"Generating for {split_name}"):
#             (log_vec, ctx_vec), _ = dataset[i]
#             log_vec, ctx_vec = log_vec.unsqueeze(0).to(DEVICE), ctx_vec.unsqueeze(0).to(DEVICE)

#             _, logits, _, _ = model(log_vec, ctx_vec)
#             probs = F.softmax(logits, dim=1)
#             confidence, _ = torch.max(probs, dim=1)
            
#             if confidence.item() < args.threshold:
#                 llm_call_count += 1

#                 target_log = raw_logs[i]
#                 start_idx = max(0, i - args.llm_context_window)
#                 context_logs = raw_logs[start_idx:i]
#                 debate_text = call_llm_for_debate(target_log, context_logs, llm_tokenizer, llm_model)
#             else:
#                 debate_text = "High confidence case."
            
#             debate_texts.append(debate_text)
    
#     print(f"LLM was called {llm_call_count} times ({llm_call_count/len(dataset)*100:.2f}% of the data).")

#     print("[4/4] Saving new dataset...")
#     df = pd.read_csv(input_csv)
#     df['debate_text'] = debate_texts
#     df.to_csv(output_csv, index=False)
#     print(f"Successfully saved {len(df)} rows to {output_csv}")


def generate_features_for_split(args, llm_tokenizer, llm_model):
    """生成平衡的辩论特征"""
    split_name = args.split
    DATASET = args.dataset
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NORMAL_LABEL = 0 # 0 为 Normal
    random.seed(42)

    input_csv = f"./datasets/{DATASET}/log_{split_name}_1p0.csv"
    if split_name == 'test':
        input_csv = f"./datasets/{DATASET}/log_test.csv"
    if DATASET == 'Tbird':
         input_csv = f"./datasets/{DATASET}/log_{split_name}_1p0.csv"
         
    output_csv = f"./datasets/{DATASET}/{split_name}_debate_balanced.csv"
    
    if os.path.exists(output_csv):
        print(f"\n--- Skipping {split_name} split (output file already exists: {output_csv}) ---")
        return
    
    print(f"\n--- Processing {split_name} split for BALANCED debate features ---")
    

    print(f"[1/5] Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    dataset = build_dataset(input_csv, split_name)

    print("[2/5] Loading trained MLP-1 model...")
    model = CoherenceModel(input_dim=384, proj_dim=128, hidden_dim=256, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(args.mlp1_model_path, map_location=DEVICE))
    model.eval()

    print("[3/5] Pass 1: Identifying low/high confidence samples for balanced sampling...")
    loader = DataLoader(dataset, batch_size=args.llm_context_window * 4, shuffle=False, num_workers=0)

    all_confidences = []
    with torch.no_grad():
        for (log_vec, ctx_vec), _ in tqdm(loader, desc="Getting confidences"):
            log_vec, ctx_vec = log_vec.to(DEVICE), ctx_vec.to(DEVICE)
            _, logits, _, _ = model(log_vec, ctx_vec)
            probs = F.softmax(logits, dim=1)
            confidence, _ = torch.max(probs, dim=1)
            all_confidences.extend(confidence.cpu().tolist())
    
    labels = df['label'].values
    
    low_conf_indices = [i for i, conf in enumerate(all_confidences) if conf < args.threshold]
    
    high_conf_normal_indices = [
        i for i, (conf, label) in enumerate(zip(all_confidences, labels))
        if conf >= args.threshold and label == NORMAL_LABEL
    ]
    
    num_low_conf = len(low_conf_indices)
    num_high_conf_candidates = len(high_conf_normal_indices)

    print(f"Found {num_low_conf} low-confidence samples (Group L).")
    print(f"Found {num_high_conf_candidates} high-confidence normal samples (candidates for sampling).")

    if num_high_conf_candidates < num_low_conf:
        print(f"Warning: Not enough high-conf normal samples ({num_high_conf_candidates}) to match low-conf ({num_low_conf}). Using all available.")
        sampled_high_conf_indices = high_conf_normal_indices
    else:
        sampled_high_conf_indices = random.sample(high_conf_normal_indices, num_low_conf)
        print(f"Randomly sampled {len(sampled_high_conf_indices)} high-conf normal samples (Group H_Normal).")

    # 合并两组，得到最终需要调用 LLM 的索引
    indices_to_debate = set(low_conf_indices + sampled_high_conf_indices)
    print(f"Total samples to call LLM for: {len(indices_to_debate)}")

    print("[4/5] Pass 2: Generating debate text features...")
    debate_texts = []
    raw_logs = df["content"].tolist()
    
    llm_call_count = 0
    for i in tqdm(range(len(raw_logs)), desc=f"Generating for {split_name}"):
        if i in indices_to_debate:
            llm_call_count += 1
            target_log = raw_logs[i]
            start_idx = max(0, i - args.llm_context_window)
            context_logs = raw_logs[start_idx:i]
            debate_text = call_llm_for_debate(target_log, context_logs, llm_tokenizer, llm_model)
        else:
            debate_text = "High confidence case."
        
        debate_texts.append(debate_text)
    
    print(f"LLM was called {llm_call_count} times ({llm_call_count/len(raw_logs)*100:.2f}% of the data).")

    # 5. 保存新的平衡数据集
    print("[5/5] Saving new balanced dataset...")
    df['debate_text'] = debate_texts
    df.to_csv(output_csv, index=False)
    print(f"Successfully saved {len(df)} rows to {output_csv}")

if __name__ == "__main__":
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

    print(f"[LLM] Loading model: {LLM_MODEL_NAME}")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME, trust_remote_code=True, device_map="auto", dtype=torch.float16
    ).eval()
    if llm_tokenizer.pad_token_id is None: llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

    # 动态设置MLP-1模型路径
    args.mlp1_model_path = f"model/{args.dataset}/{args.dataset}_model.pt"

    # 依次处理 train, val, test
    args.split = "train"
    generate_features_for_split(args, llm_tokenizer, llm_model)
    
    args.split = "val"
    generate_features_for_split(args, llm_tokenizer, llm_model)
    
    args.split = "test"
    generate_features_for_split(args, llm_tokenizer, llm_model)
