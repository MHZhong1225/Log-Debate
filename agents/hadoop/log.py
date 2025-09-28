#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-log micro-analysis with Qwen-2.5-7B-Instruct
--------------------------------------------------
• 为每条日志生成 “类别: 因果解释” 单句  
• 同时提取预测类别并保存  
• 自动输出分类报告与混淆矩阵
"""
import os
import re
import sys
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ───────────── 配置区 ─────────────
INPUT_CSV  = "./datasets/Hadoop/hadoop_csv.csv"
OUTPUT_DIR   = "./datasets/Hadoop/explanations/"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "hadoop_log_explanation.csv")
CM_PNG     = os.path.join(OUTPUT_DIR, "confusion_matrix_log.pdf")

TEST_SIZE  = None          # None = 全量
MAX_NEW_TOKENS = 384

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

VALID_LABELS = ["Normal", "Network disconnection", "Machine down", "Disk full"]

# ───────────── 初始化 ─────────────
sys.stdout.reconfigure(encoding="utf-8")


def init_model():
    print(f"[INFO] Loading model {MODEL_NAME} → {DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    mdl.generation_config = GenerationConfig.from_model_config(mdl.config)
    return tok, mdl


# ───────────── 构造 Prompt 与推理 ─────────────
def build_prompt(log_text: str) -> str:
    prompt = f"""You are a Log Forensics Specialist. Your task is to perform a micro-analysis of a single log entry, translating its technical content into a clear, causal narrative.

    [Log Category Definitions]:
    The valid log categories are exactly: Normal, Network disconnection, Machine down, Disk full.
    **Assign exactly one class from this list.**
    Your output must exactly match one of these categories. 

    [Log Classification Basis]
    Normal:
    Contains routine job execution terms like taskattempt, progress, maptask, buffer, mapreduce, and date stamps (2015-10-17, 2015-10-18). Mentions of initiated and id suggest successful, uninterrupted operations.

    Disk full:
    Frequent occurrence of terms like taskattempt, maptask, mapreduce, buffer, key-value, and completion, often alongside data indexing or progress updates. Mentions of resource-intensive operations such as starting, events, and hadoop processing steps.

    Machine down:
    Similar MapReduce workflow terms (taskattempt, maptask, map, completion), but often coupled with specific dates (2015-10-17), high numeric thresholds (10000), and “reached” events, indicating task failures or node unavailability.

    Network disconnection:
    Presence of networking and Java-related terms such as connection, client, ipc, invocation, lease, failed, method, line, class, apache, and client.java. These highlight communication failures between Hadoop components.

    [Key Differentiation Criteria Between Network Disconnection, Machine Down, Disk Full, and Normal Logs]
    | Category                | Key Keywords / Signals                                                         | Main Differentiation Points                                                 |
    | ----------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
    | Normal                | taskattempt, progress, maptask, buffer, key-value, mapreduce, initiated          | Routine job execution, normal progress updates, no errors or failures       |
    | Network Disconnection | connection, client, ipc, invocation, lease, failed, method, line, class, apache  | Communication failures, Hadoop IPC issues, socket/connection errors, timeouts|
    | Machine Down          | taskattempt, maptask, attempt, reached, 10000, completion, starting, hadoop      | Node/machine unavailability, task termination without disk or network errors |
    | Disk Full             | taskattempt, maptask, mapreduce, buffer, key-value, completion, starting, events | Disk space exhaustion, failed write operations, no network-related exceptions|


    [Task]:
    Your response must be a **single, information-dense sentence in English** that accomplishes two things:
    1.  Begin with the determined log category (e.g., "Normal:").
    2.  Follow with a concise narrative that explains the **cause and effect** of the event described in this specific log. Focus only on what can be inferred from this single entry.

    **Translate the technical log into a story of what happened.**

    [Raw Log Content]:
    {log_text}
    """
    return prompt


def call_qwen(tok, mdl, log_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are Qwen, an expert system-log analysis assistant."},
        {"role": "user",   "content": build_prompt(log_text)}
    ]
    chat_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs    = tok(chat_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out_ids = mdl.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    new_ids  = out_ids[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True).strip() or "Empty response"


def extract_label(sentence: str) -> str:
    """取句首第一个有效标签"""
    for lbl in VALID_LABELS:
        if re.match(fr"^\s*{re.escape(lbl)}\b", sentence, flags=re.IGNORECASE):
            return lbl
    return "Unknown"


# ───────────── 主流程 ─────────────
def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    df = (pd.read_csv(INPUT_CSV)
            .dropna(subset=["content"])
            .reset_index(drop=True))

    if TEST_SIZE is not None:
        df = df.sample(n=min(TEST_SIZE, len(df)), random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampled {len(df)} rows")

    tok, mdl = init_model()

    explanations, preds = [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="LLM inference"):
        try:
            exp = call_qwen(tok, mdl, row["content"])
        except RuntimeError as e:
            print(f"\n[WARN] GPU OOM at row {idx}: {e}. Retrying on CPU …")
            mdl.to("cpu")
            exp = call_qwen(tok, mdl, row["content"])
            mdl.to(DEVICE)
        except Exception as e:
            print(f"\n[ERROR] Row {idx} failed: {e}")
            exp = "Generation failed."

        explanations.append(exp)
        preds.append(extract_label(exp))

    df["predicted_label"] = preds
    df["explanation"]     = explanations
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[DONE] Saved predictions & explanations → {OUTPUT_CSV}")

    # ─── 评估报告 ───
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(classification_report(
        df["label"], df["predicted_label"],
        labels=VALID_LABELS,
        zero_division=0
    ))

    cm   = confusion_matrix(df["label"], df["predicted_label"], labels=VALID_LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=VALID_LABELS)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix (log)")
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=300)
    print(f"[INFO] Confusion matrix saved → {CM_PNG}")


if __name__ == "__main__":
    main()
