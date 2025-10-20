import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import argparse
import random
import re # 用于解析 LLM 输出
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from dataset_log_with_context import LogWithContextDataset
from context_agent import ContextAgent
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import CoherenceModel as MLP1Model

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the RCDL (Risk-Controllable Debate Layer) Model")
    p.add_argument("--dataset",      default="BGL", help="Dataset name")
    p.add_argument("--batch_size",   type=int,   default=256, help="Batch size for evaluation")
    p.add_argument("--context_window", type=int, default=5, help="Number of preceding logs for LLM context")
    
    # RCDL 关键参数
    p.add_argument("--lambda_coeff", "-l", type=float, default=0.5, help="Weight for LLM risk score r_t (lambda)")
    p.add_argument("--decision_threshold", "-d", type=float, default=0.5, help="Final decision threshold (delta)")
    
    # 触发 LLM 的不确定性区间
    # 默认值 [0.1, 0.9] 意味着对非常自信的样本 (s_t < 0.1 或 s_t > 0.9) 跳过 LLM
    p.add_argument("--uncertainty_lower", type=float, default=0.08, help="Lower bound of uncertainty (trigger LLM if s_t > this)")
    p.add_argument("--uncertainty_upper", type=float, default=0.9, help="Upper bound of uncertainty (trigger LLM if s_t < this)")
    
    return p.parse_args()

# def call_llm_for_debate_RCDL(target_log: str, context_logs: list[str], mlp1_score: float, tokenizer, model) -> str:
def call_llm_for_debate_RCDL(target_log: str, context_logs: list[str], tokenizer, model) -> str:
    """
    调用 LLM 进行 RCDL 辩论，并要求提供风险分数。
    """
    context_str = "\n".join(f"- {log}" for log in context_logs) if context_logs else "No preceding context available."

    # The system's initial coherence model scored this log's anomaly probability at: {mlp1_score:.4f}
    # Risk Score (0.0 to 1.0):
    # (Provide *only* a single numeric value representing your final risk assessment.)
    # (Provide 1-2 concise bullet points arguing why the Target Log could be anomalous.)

    prompt = f"""
As an expert Site Reliability Engineer, your task is to conduct a structured, evidence-based debate on whether the "Target Log" is anomalous, based on the "Context".
(A score near 0.0 is Normal, a score near 1.0 is Anomalous)

### Inputs:
[Target Log to Analyze]:
{target_log}

[Context]:
{context_str}

### Structured Debate Output:
Evidence for Anomaly:
(Provide 1 concise bullet points that best argues why the Target Log could be anomalous.)

Evidence for Normality:
(Provide 1 concise bullet points that best argues why the Target Log could be normal.)

Final Verdict:
Use your reasoning to reach a definite conclusion, decide which explanation is more convincing, provide the final verdict. (You MUST classify the "Target Log" as either ANOMALY or NORMAL. Respond with *ONLY* the single word "ANOMALY" or "NORMAL" below this line.)
"""
    messages = [{"role": "system", "content": "You are an expert system engineer responsible for log anomaly analysis."}, {"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False) # 使用 do_sample=False 保证分数稳定性
        
    out_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    return out_text or "Analysis could not be generated."

def parse_risk_score(llm_output: str, default_score: float) -> float:
    """
    从 LLM 的完整输出中解析出最后的风险分数。
    """
    try:
        # 寻找 "Risk Score (0.0 to 1.0):" 之后的内容
        match = re.search(r"Risk Score \(0\.0 to 1\.0\):.*", llm_output, re.DOTALL | re.IGNORECASE)
        if not match:
            # 备用方案：寻找 "Risk Score:"
             match = re.search(r"Risk Score:.*", llm_output, re.DOTALL | re.IGNORECASE)

        if match:
            # 提取匹配后的文本
            score_text = match.group(0).split(":")[-1].strip()
            # 再次使用 regex 提取文本中的第一个浮点数
            score_match = re.search(r"(\d\.\d+)", score_text)
            if score_match:
                return float(score_match.group(1))

        # 如果找不到, 尝试直接在整个文本中寻找
        score_match = re.search(r"(\d\.\d+)", llm_output)
        if score_match:
            return float(score_match.group(1))

        print(f"\n[Warning] Could not parse risk score from LLM output. Defaulting to MLP-1 score. Output:\n{llm_output}")
        return default_score
    except Exception as e:
        print(f"\n[Error] Error parsing risk score: {e}. Defaulting to MLP-1 score. Output:\n{llm_output}")
        return default_score

# 在 4_evaluate_RCDL.py 中替换此函数
def parse_llm_verdict(llm_output: str, default_score_s_t: float) -> float:
    """
    (V2) 解析 LLM 输出中的 "ANOMALY" 或 "NORMAL" 判决。
    返回 1.0 (Anomaly) 或 0.0 (Normal)。
    如果无法解析，返回原始的 s_t 分数，以安全地“取消”融合操作。
    """
    try:
        # 优先在 "Final Verdict:" 之后寻找
        match = re.search(r"Final Verdict:.*", llm_output, re.DOTALL | re.IGNORECASE)
        text_to_search = llm_output
        if match:
            text_to_search = match.group(0)

        # 查找关键词
        if re.search(r"\bANOMALY\b", text_to_search, re.IGNORECASE):
            return 1.0  # LLM 认为是异常
        if re.search(r"\bNORMAL\b", text_to_search, re.IGNORECASE):
            return 0.0  # LLM 认为是正常

        print(f"\n[Warning] V2: Could not parse ANOMALY/NORMAL. Defaulting to s_t. Output:\n{llm_output}")
        return default_score_s_t
    except Exception as e:
        print(f"\n[Error] V2: Error parsing verdict: {e}. Defaulting to s_t. Output:\n{llm_output}")
        return default_score_s_t

def main():
    args = parse_args()
    DATASET = args.dataset
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NORMAL_LABEL = 0

    # --- 1. 定义模型路径 ---
    MLP1_MODEL_PATH = f"model/{DATASET}/{DATASET}_model.pt"
    if not os.path.exists(MLP1_MODEL_PATH):
        print(f"Error: MLP-1 model not found at {MLP1_MODEL_PATH}. Please run 1_train_mlp.py first.")
        return
        
    # LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    LLM_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

    # --- 2. 加载 LLM ---
    print(f"[LLM] Loading model: {LLM_MODEL_NAME}")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME, trust_remote_code=True, device_map="auto", dtype=torch.float16
    ).eval()
    if llm_tokenizer.pad_token_id is None: llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

    # --- 3. 加载 MLP-1 (守门员/CCF) ---
    print("Loading trained Coco...")
    model_mlp1 = MLP1Model(input_dim=384, proj_dim=128, hidden_dim=256, num_classes=2).to(DEVICE)
    model_mlp1.load_state_dict(torch.load(MLP1_MODEL_PATH, map_location=DEVICE))
    model_mlp1.eval()

    # --- 4. 加载测试数据 (仅使用 MLP-1 的 Dataset) ---
    print("--- Loading Full Test Data for RCDL Evaluation ---")
    
    test_csv = f"./datasets/{DATASET}/log_test.csv"
    if DATASET == 'Tbird':
        test_csv = f"./datasets/{DATASET}/log_test_1p0.csv"
        
    df = pd.read_csv(test_csv)
    raw_logs = df["content"].tolist() # 我们需要原始日志文本来喂给 LLM
    
    # 我们只使用 LogWithContextDataset 来获取 log_vec 和 ctx_vec
    mlp1_test_ds = LogWithContextDataset(
        raw_logs=raw_logs, 
        labels=df["label"].tolist(), 
        mode='inference', 
        ctx_agent=ContextAgent()
    )
    
    test_loader = DataLoader(mlp1_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- 5. 执行 RCDL 融合评估 ---
    print(f"--- Running RCDL Evaluation ---")
    print(f"Uncertainty Band: ({args.uncertainty_lower}, {args.uncertainty_upper})")
    print(f"Fusion Lambda (LLM weight): {args.lambda_coeff}")
    print(f"Final Decision Threshold: {args.decision_threshold}")
    
    y_true_all = []
    y_pred_all = []
    llm_call_count = 0
    
    all_final_scores = []
    all_true_labels = []

    with torch.no_grad():
        for i, ((log_vec, ctx_vec), labels) in enumerate(tqdm(test_loader, desc="RCDL Evaluation")):
            
            log_vec = log_vec.to(DEVICE)
            ctx_vec = ctx_vec.to(DEVICE)
            labels_np = labels.cpu().numpy()

            # Coco
            _, logits_mlp1, _, _ = model_mlp1(log_vec, ctx_vec)
            anomaly_probs_mlp1 = F.softmax(logits_mlp1, dim=1)[:, 1]
            final_anomaly_scores = anomaly_probs_mlp1.clone()

            # [ , ]
            trigger_mask = (anomaly_probs_mlp1 >= args.uncertainty_lower) & \
                           (anomaly_probs_mlp1 <= args.uncertainty_upper)
            
            # debate
            if trigger_mask.any():
                triggered_indices = trigger_mask.nonzero(as_tuple=False).squeeze(1)
                
                for idx_in_batch in triggered_indices:
                    llm_call_count += 1
                    global_idx = i * args.batch_size + idx_in_batch.item()
                    s_t = anomaly_probs_mlp1[idx_in_batch].item()
                    
                    # 准备 LLM 输入
                    target_log = raw_logs[global_idx]
                    start_ctx_idx = max(0, global_idx - args.context_window)
                    context_logs = raw_logs[start_ctx_idx:global_idx]
                    
                    llm_output = call_llm_for_debate_RCDL(target_log, context_logs, llm_tokenizer, llm_model)
                    print(llm_output)
                    #    如果解析失败, r_t = s_t, 融合自动取消 (p_t = s_t)
                    r_t = parse_llm_verdict(llm_output, default_score_s_t=s_t)
                    
                    # (p_t)
                    p_t = (args.lambda_coeff * r_t) + (1 - args.lambda_coeff) * s_t

                    final_anomaly_scores[idx_in_batch] = p_t


                    # # 调用 LLM
                    # llm_output = call_llm_for_debate_RCDL(target_log, context_logs, s_t, llm_tokenizer, llm_model)
                    # print(llm_output)

                    # # (r_t) - 解析 LLM 的风险分数
                    # r_t = parse_risk_score(llm_output, default_score=s_t)
                    
                    # # (p_t) - 融合
                    # p_t = (args.lambda_coeff * r_t) + (1 - args.lambda_coeff) * s_t
                    
                    # # 更新最终分数
                    # final_anomaly_scores[idx_in_batch] = p_t

            # 4. 根据最终分数和决策阈值 (delta) 得到预测
            final_preds = (final_anomaly_scores > args.decision_threshold).int()
            
            # 转换为二分类 (0=Normal, 1=Anomaly)
            y_true_binary = (labels_np != NORMAL_LABEL).astype(int)
            y_pred_binary = final_preds.cpu().numpy()
            
            y_true_all.extend(y_true_binary)
            y_pred_all.extend(y_pred_binary)

    # --- 6. 打印最终报告 ---
    print("\n--- Final RCDL Model Evaluation Report ---")
    print(f"Total Samples: {len(y_true_all)}")
    print(f"Called LLM for {llm_call_count} 'uncertain' samples ({llm_call_count/len(y_true_all)*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_true_all, y_pred_all, target_names=['Normal', 'Anomaly'], digits=4))

if __name__ == "__main__":
    main()