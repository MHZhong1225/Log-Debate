#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log‐level anomaly classification with Qwen-2.5-7B-Instruct
=========================================================
1. 逐条日志抽取“小时时间戳”并构建同小时上下文；
2. 利用大语言模型 Qwen 给出带上下文的单句判定与解释；
3. 评估预测表现并输出混淆矩阵与详细结果。
"""
import os
import re
from typing import List

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ──────────────────────────────────────────────────────────
# 配置区
# ──────────────────────────────────────────────────────────
# 数据路径
INPUT_CSV  = "./datasets/Hadoop/hadoop_csv.csv"
OUTPUT_DIR   = "./datasets/Hadoop/explanations/"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "hadoop_context_explanation.csv")

# 上下文条数
CONTEXT_K  = 10
# 取多少条做 Demo / 评估，如需全量请改为 None
TEST_SIZE  = None       # None 表示全量

# LLM 参数
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 128

# 分类标签
VALID_LABELS = ["Normal", "Network disconnection", "Machine down", "Disk full"]

# ──────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────
def extract_log_hour(content: str) -> str:
    """提取日志中的‘年.月.日 月缩写 日 时’—用作同小时分组键"""
    try:
        date_match = re.search(r'(\d{4}\.\d{2}\.\d{2})', content)
        hour_match = re.search(r'([A-Z][a-z]{2}) (\d{1,2}) (\d{2}):', content)
        if date_match and hour_match:
            date_part  = date_match.group(1)
            month_abbr = hour_match.group(1)
            day        = hour_match.group(2).zfill(2)
            hour       = hour_match.group(3).zfill(2)
            return f"{date_part} {month_abbr} {day} {hour}"
    except Exception:
        pass
    return "unknown_time"


def load_with_hourly_context(csv_path: str, k: int = 10) -> pd.DataFrame:
    """读取日志并为每条记录构造同小时上下文"""
    print("[INFO] Loading raw data …")
    df = pd.read_csv(csv_path).dropna(subset=["content", "label"]).reset_index(drop=True)

    print("[INFO] Extracting hour keys …")
    df["hour_key"]  = df["content"].apply(extract_log_hour)
    df["unique_id"] = np.arange(len(df))  # 用于排除自身

    print("[INFO] Grouping logs by hour …")
    hourly_groups = df.groupby("hour_key")

    contexts: List[str] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building hourly context"):
        hour_key   = row["hour_key"]
        current_id = row["unique_id"]

        if hour_key == "unknown_time" or hour_key not in hourly_groups.groups:
            contexts.append("无有效的同小时上下文")
            continue

        other_logs = hourly_groups.get_group(hour_key)
        other_logs = other_logs[other_logs["unique_id"] != current_id]

        if other_logs.empty:
            contexts.append("该小时内无其他日志")
            continue

        sampled = other_logs.sample(n=min(k, len(other_logs)), random_state=42)
        context_text = "\n---\n".join(
            f"[{r['label']}] 日志: {r['content']}" for _, r in sampled.iterrows()
        )
        contexts.append(context_text)

    df["context"] = contexts
    return df.drop(columns=["hour_key", "unique_id"])


def extract_label_from_sentence(sentence: str) -> str:
    """从 LLM 单句输出的句首提取有效标签"""
    for lbl in VALID_LABELS:
        if re.match(fr"^\s*{re.escape(lbl)}\b", sentence, flags=re.IGNORECASE):
            return lbl
    return "未知"


# ──────────────────────────────────────────────────────────
# LLM 载入
# ──────────────────────────────────────────────────────────
def init_model(model_name: str = MODEL_NAME):
    print(f"[INFO] Loading model {model_name} to {DEVICE} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config = GenerationConfig.from_model_config(model.config)
    return tokenizer, model


def call_qwen_with_context(tokenizer, model, log: str, context: str) -> str:
    prompt = f"""You are a System Behavior Analyst. Your task is to assess a target log's behavior by comparing it against the backdrop of other logs from the same hour to identify anomalies or confirm system-wide trends.

    [Log Category Definitions]:
    The valid log categories are exactly: Normal, Network disconnection, Machine down, Disk full.
    **Assign exactly one class from this list.**

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
    | Category                  | Key Keywords / Signals                                                           | Main Differentiation Points                                                   |
    | ------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
    | **Normal**                | taskattempt, progress, maptask, buffer, key-value, mapreduce, initiated          | Routine job execution, normal progress updates, no errors or failures         |
    | **Network Disconnection** | connection, client, ipc, invocation, lease, failed, method, line, class, apache  | Communication failures, Hadoop IPC issues, socket/connection errors, timeouts |
    | **Machine Down**          | taskattempt, maptask, attempt, reached, 10000, completion, starting, hadoop      | Node/machine unavailability, task termination without disk or network errors  |
    | **Disk Full**             | taskattempt, maptask, mapreduce, buffer, key-value, completion, starting, events | Disk space exhaustion, failed write operations, no network-related exceptions |

    ---
    [Judgment Principles for Contextual Analysis]:
    When comparing the target log with its peers, focus on:
    1.  **Behavior Deviation**: Is the target log an outlier compared to a mostly normal baseline?
    2.  **Co-occurrence Trend**: Is the target log part of a larger pattern of similar errors or events happening concurrently?

    [Your Task]:
    Your response must be a **single, information-dense sentence in English** that accomplishes two things:
    1.  Begin with the determined log category (e.g., "Network disconnection:").
    2.  Follow with a concise justification that is **explicitly based on the contextual comparison**. You must state whether your conclusion is based on the log being an anomaly (Behavior Deviation) or part of a trend (Co-occurrence Trend).

    **Focus on whether this log is an outlier or part of a larger pattern.**

    [Target Log]:
    {log}

    [Other Logs from the Same Hour]:
    {context}
    """

    messages = [
        {"role": "system", "content": "You are Qwen, an expert system-log analysis assistant."},
        {"role": "user",   "content": prompt}
    ]

    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs    = tokenizer(chat_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids     = output_ids[0][inputs["input_ids"].shape[1]:]
    out_text    = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return out_text or "Empty response"


# ──────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_full = load_with_hourly_context(INPUT_CSV, k=CONTEXT_K)


    if TEST_SIZE is not None:
        df = df_full.sample(n=min(TEST_SIZE, len(df_full)), random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampled {len(df)} rows for evaluation.")
    else:
        df = df_full

    tokenizer, model = init_model()

    predictions, explanations = [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="LLM inference"):
        try:
            explanation = call_qwen_with_context(
                tokenizer, model,
                log=row["content"],
                context=row["context"]
            )
        except RuntimeError as e:   # 处理显存溢出等异常
            print(f"\n[WARN] GPU error on row {idx}: {e}. Retrying on CPU …")
            model.to("cpu")
            explanation = call_qwen_with_context(
                tokenizer, model,
                log=row["content"],
                context=row["context"]
            )
            model.to(DEVICE)
        # print(f"[0 Label]: {row['label']}")
        # print(f"[1 Reasoning]: {explanation}")
        pred_label = extract_label_from_sentence(explanation)
        predictions.append(pred_label)
        explanations.append(explanation)

    df["predicted_label"]   = predictions
    df["context_explanation"] = explanations
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[INFO] Results saved to {OUTPUT_CSV}")

    # ── 评估 ──────────────────────────────────────────────
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
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_context.pdf")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    print(f"[INFO] Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
