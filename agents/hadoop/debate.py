#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log-level Anomaly Adjudication with Qwen-2.5-7B-Instruct
=========================================================
1.  合并“单日志分析”与“上下文分析”的结果；
2.  利用 LLM (Qwen) 作为“裁判”进行最终裁决与解释生成；
3.  评估“裁判”模型的分类表现并输出结果。
"""
import os
import re
import json
from typing import Dict, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ──────────────────────────────────────────────────────────
# 配置区 (Configuration)
# ──────────────────────────────────────────────────────────
# 数据路径
DATA_DIR   = "./datasets/Hadoop/explanations/"
OUTPUT_DIR = "./datasets/Hadoop/results/"
LOG_CSV    = os.path.join(DATA_DIR, "hadoop_log_explanation.csv")
CTX_CSV    = os.path.join(DATA_DIR, "hadoop_context_explanation.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "hadoop_debate.csv")
METRICS_TXT = os.path.join(OUTPUT_DIR, "debate_agent_metrics.txt")

# LLM 参数
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256

# 分类标签
VALID_LABELS = ["Normal", "Network disconnection", "Machine down", "Disk full"]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ──────────────────────────────────────────────────────────
# LLM 与工具函数 (LLM & Utilities)
# ──────────────────────────────────────────────────────────
def init_model(model_name: str = MODEL_NAME) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """加载并初始化分词器和 LLM 模型"""
    print(f"[INFO] Loading model {model_name} to {DEVICE} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.generation_config.cache_implementation = "static"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config = GenerationConfig.from_model_config(model.config)
    return tokenizer, model


def call_debate_agent(
    tokenizer: AutoTokenizer, model: AutoModelForCausalLM,
    content: str, log_agent_exp: str, context_agent_exp: str
) -> str:
    prompt = f"""You are the Lead Adjudicator for a team of AI log security analysts. Your primary role is to resolve and synthesize the findings from your two specialist agents to produce a final, authoritative judgment on a log's category.
[Log Category Definitions]:
The valid log categories are exactly: Normal, Network disconnection, Machine down, Disk full.

[Your Specialist Agents' Reports]:
- **Log Agent Report (Micro-Analysis)**: Focuses on the intrinsic, causal story of the single log entry.
- **Context Agent Report (Macro-Analysis)**: Focuses on the log's behavior relative to its peers in the same hour (anomaly or trend).

---
[CASE FILE]

[Raw Log Content]:
{content}

[Log Agent Report]:
{log_agent_exp}

[Context Agent Report]:
{context_agent_exp}

---
[YOUR ADJUDICATION TASK]:
1.  **Assess Agreement**: First, compare the conclusions from your two agents.
2.  **Synthesize or Debate**:
    -   **If Agents Agree**: Your task is to **synthesize**. Weave together the micro-analysis (the "what") and the macro-analysis (the "so what") into a single, definitive narrative that is stronger than either report alone.
    -   **If Agents Disagree**: Your task is to **adjudicate the debate**. Analyze the conflicting evidence, decide which agent's reasoning is more compelling (or if a third category is more appropriate), and your explanation **must justify your final decision** over the rejected one.

[OUTPUT REQUIREMENTS]:
Your response must be a JSON object with two keys: "category" and "explanation".

[Example of High-Quality Explanation Style]:
{{
  "category": "Network disconnection",
  "explanation": "This log narrates a client's struggle to connect to a server. The key process is an 'ipc.Client', which is actively in a 'Retrying connect' loop. This indicates the client itself is operational but is failing to establish a network pathway, pointing to an Inter-Process Communication failure."
}}
"""
    messages = [
        {"role": "system", "content": "You are Qwen, an expert system log analysis assistant."},
        {"role": "user", "content": prompt}
    ]

    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    out_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return out_text or "Empty response"


def extract_label_from_response(response: str) -> Tuple[str, str]:
    """从 LLM 的响应中提取分类标签和解释，优先解析 JSON"""
    try:
        # 提取被 ```json ... ``` 包围的内容
        json_match = re.search(r'```json\s*([\s\S]+?)\s*```', response)
        if json_match:
            json_str = json_match.group(1)
        else: # 如果没有 markdown 标记，则假定整个响应是 JSON
            json_str = response
        
        data = json.loads(json_str)
        category = data.get("category", "未知")
        explanation = data.get("explanation", response)
        
        if category in VALID_LABELS:
            return category, explanation
        return "未知", explanation

    except (json.JSONDecodeError, AttributeError):
        # Fallback: 如果 JSON 解析失败，则在整个响应文本中正则搜索标签
        print(f"[WARN] JSON parsing failed for: {response}. Falling back to regex search.")
        for label in VALID_LABELS:
            if re.search(r'\b' + re.escape(label) + r'\b', response, re.IGNORECASE):
                return label, response
        return "未知", response


# ──────────────────────────────────────────────────────────
# 数据处理与评估 (Data Handling & Evaluation)
# ──────────────────────────────────────────────────────────
def load_and_prepare_data(log_path: str, ctx_path: str) -> pd.DataFrame:
    """加载、合并和验证数据"""
    print("[INFO] Loading and merging datasets …")
    log_df = pd.read_csv(log_path).rename(columns={"explanation": "log_explanation"})
    ctx_df = pd.read_csv(ctx_path).rename(columns={"context_explanation": "context_explanation"})
    
    # 确保用于合并的 ID 列类型一致
    log_df['id'] = log_df['id'].astype(str)
    ctx_df['id'] = ctx_df['id'].astype(str)

    # 选择需要的列进行合并
    merged_df = pd.merge(
        log_df[['id', 'content', 'log_explanation', 'label']],
        ctx_df[['id', 'context_explanation']],
        on='id'
    )
    
    print(f"[INFO] Merged data shape: {merged_df.shape}")
    return merged_df.sample(frac=1, random_state=42).reset_index(drop=True)


def evaluate_and_save(df: pd.DataFrame, metrics_path: str) -> None:
    """计算分类指标并保存报告"""
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)

    y_true = df["label"]
    y_pred = df["predicted_label"]
    
    # 打印详细报告
    report = classification_report(
        y_true, y_pred, labels=VALID_LABELS, zero_division=0, digits=4
    )
    print(report)

    # 计算宏平均指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    summary = (
        f"Overall Metrics:\n"
        f"  Accuracy:         {acc:.4f}\n"
        f"  Macro Precision:  {prec:.4f}\n"
        f"  Macro Recall:     {rec:.4f}\n"
        f"  Macro F1-Score:   {f1:.4f}\n"
    )
    print(summary)
    
    # 将报告和汇总指标写入文件
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("Classification Report:\n")
        f.write("======================\n")
        f.write(report)
        f.write("\n\nSummary Metrics:\n")
        f.write("================\n")
        f.write(summary)
    print(f"[INFO] Classification metrics saved to {metrics_path}")


# ──────────────────────────────────────────────────────────
# 主流程 (Main Workflow)
# ──────────────────────────────────────────────────────────
def main() -> None:
    """主执行函数"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_prepare_data(LOG_CSV, CTX_CSV)
    tokenizer, model = init_model()

    predictions, explanations = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM Adjudication"):
        try:
            decision = call_debate_agent(
                tokenizer, model,
                content=row["content"],
                log_agent_exp=row["log_explanation"],
                context_agent_exp=row["context_explanation"]
            )
        except RuntimeError as e:
            print(f"\n[WARN] GPU error: {e}. Retrying on CPU for this sample…")
            torch.cuda.empty_cache()
            model.to("cpu")
            decision = call_debate_agent(
                tokenizer, model,
                content=row["content"],
                log_agent_exp=row["log_explanation"],
                context_agent_exp=row["context_explanation"]
            )
            model.to(DEVICE)
        
        pred_label, explanation = extract_label_from_response(decision)
        predictions.append(pred_label)
        explanations.append(explanation)

    df["predicted_label"] = predictions
    df["final_explanation"] = explanations
    
    # 保存详细结果
    cols_to_save = ['id', 'label', 'predicted_label', 'final_explanation', 'content']
    df[cols_to_save].to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\n[INFO] Inference results saved to {OUTPUT_CSV}")

    # 评估并保存指标
    evaluate_and_save(df, METRICS_TXT)


if __name__ == "__main__":
    main()