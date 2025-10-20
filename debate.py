# debate.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
import torch.nn.functional as F

# 导入你用于训练双塔模型的新版 Dataset
from dataset_log_with_context_twotower import LogWithContextDataset
from context_agent import ContextAgent

# 导入 LLM 相关库
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ================== 1. 定义双塔模型 (必须与训练时完全一致) ==================
class TwoMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        tower_input_dim = input_dim
        self.tower_output_dim = hidden_dim // 2 # 记录塔的输出维度

        self.log_tower = nn.Sequential(
            nn.Linear(tower_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.tower_output_dim)
        )
        self.context_tower = nn.Sequential(
            nn.Linear(tower_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.tower_output_dim)
        )
        fused_dim = self.tower_output_dim * 2
        self.classifier_head = nn.Linear(fused_dim, num_classes)

    def forward(self, log_vec, ctx_vec):
        log_features = self.log_tower(log_vec)
        ctx_features = self.context_tower(ctx_vec)
        fused_features = torch.cat([log_features, ctx_features], dim=-1)
        logits = self.classifier_head(fused_features)
        return logits

# ================== 配置区 ==================
DATASET = "HDFS" # BGL HDFS Tbird Hadoop 
BATCH_SIZE = 32 # 仅用于数据加载
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用你训练好的双塔模型
TWOTOWER_MODEL_PATH = f"{DATASET}_best_twotower_mlp_mc.pt"
TEST_CSV = f"./datasets/{DATASET}/log_test.csv"

# LLM 配置
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 256

CLASS_MAP = {0: "Normal", 1: "Anomaly"}
if DATASET == 'Hadoop':
    CLASS_MAP = {0: "Normal", 1: "Network Disconnection", 2: "Machine Down", 3: "Disk Full"}
CONFIDENCE_THRESHOLD = 0.70 

# ================== LLM 辩论函数 (与之前版本相同) ==================
print(f"[LLM] Loading model: {LLM_MODEL_NAME}")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    dtype=torch.float16
).eval()
if llm_tokenizer.pad_token_id is None:
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

def call_debate_llm(log_content: str, log_agent_pred: str, context_agent_pred: str, reason: str, confidence: float) -> str:
    prompt = f"""You are the Lead Adjudicator for a team of AI log security analysts. Your task is to resolve a complex case identified by an advanced analysis model.

[Log Category Definitions]:
The valid log categories are: {', '.join(CLASS_MAP.values())}.

[Reason for Adjudication]: {reason}
[Model's Confidence Score]: {confidence:.2f} (A low score indicates uncertainty)

[The Analysis]:
- **Log**: Analyzed the log in isolation and concluded its category is "{log_agent_pred}".
- **Context**: Analyzed the log's surrounding context and concluded its category is "{context_agent_pred}".

---
[CASE FILE]
[Raw Log Content]:
{log_content}

---
[YOUR ADJUDICATION TASK]:
Analyze the raw log content, consider the analysis from the two towers, and make a final, justified decision.

[OUTPUT REQUIREMENTS]:
Your response must be a JSON object with a single key: "final_category".
"""
    messages = [{"role": "system", "content": "You are an expert system log analysis adjudicator."}, {"role": "user", "content": prompt}]
    chat_text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer(chat_text, return_tensors="pt").to(llm_model.device)
    with torch.no_grad():
        output_ids = llm_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, eos_token_id=llm_tokenizer.eos_token_id)
    out_text = llm_tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        import json
        data = json.loads(out_text)
        final_category = data.get("final_category", "Normal")
        if final_category in CLASS_MAP.values(): return final_category
    except:
        for cat in CLASS_MAP.values():
            if cat in out_text: return cat
    return "Normal"

def build_dataset(csv_path, split_name):
    df = pd.read_csv(csv_path)
    logs   = df["content"].tolist()
    labels = df["label"].tolist()
    ctx_agent = ContextAgent(window_size=100, hist_agg="max")
    ds = LogWithContextDataset(
        raw_logs=logs,
        labels=labels,
        mode=split_name,
        ctx_agent=ctx_agent
    )
    return ds, np.array(labels)

# ================== 主流程 ==================

# 1. 加载测试数据
print("[Data] Loading test dataset for Two-Tower model...")
test_ds, y_true = build_dataset(TEST_CSV, "inference")


# 2. 构建并加载双塔模型f
NUM_CLASSES = len(CLASS_MAP)
INPUT_DIM = 384 # 假设 all-MiniLM-L6-v2 输出维度
model = TwoMLP(
    input_dim=INPUT_DIM,
    hidden_dim=256,
    num_classes=NUM_CLASSES
).to(DEVICE)
model.load_state_dict(torch.load(TWOTOWER_MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"[Model] Two-Tower model loaded from {TWOTOWER_MODEL_PATH}")

#  3
final_preds = []
debated_cases = 0
raw_logs = pd.read_csv(TEST_CSV)["content"].tolist()
reverse_class_map = {v: k for k, v in CLASS_MAP.items()}

with torch.no_grad():
    # 我们可以使用 DataLoader 来加速 MLP 的推理过程
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    all_final_preds = []
    all_final_confidences = []

    print("[Phase 1] Getting model's primary predictions and confidences...")
    for (log_vec, ctx_vec), _ in tqdm(test_loader, desc="Model Inference"):
        log_vec = log_vec.to(DEVICE)
        ctx_vec = ctx_vec.to(DEVICE)

        # --- 直接获取完整模型的最终意见和置信度 ---
        logits = model(log_vec, ctx_vec)
        probs = F.softmax(logits, dim=1)
        confidence, pred_label = torch.max(probs, dim=1)
        
        all_final_preds.extend(pred_label.cpu().tolist())
        all_final_confidences.extend(confidence.cpu().tolist())

    print("\n[Phase 2] Adjudicating low-confidence cases with LLM...")
    for i in tqdm(range(len(all_final_preds)), desc="LLM Adjudication"):
        confidence = all_final_confidences[i]
        prediction = all_final_preds[i]
        
        # --- 决策逻辑被极大简化 ---
        # 只在最终模型的置信度低于阈值时，才启动辩论
        if confidence < CONFIDENCE_THRESHOLD:
            debated_cases += 1
            reason_for_debate = f"Model's confidence ({confidence:.2f}) is below the threshold."
            
            # 但这只用于生成 Prompt，不再作为触发条件
            (log_vec, ctx_vec), _ = test_ds[i]
            log_vec = log_vec.unsqueeze(0).to(DEVICE)
            ctx_vec = ctx_vec.unsqueeze(0).to(DEVICE)
            log_features = model.log_tower(log_vec)
            ctx_features = model.context_tower(ctx_vec)
            fused_for_log = torch.cat([log_features, torch.zeros_like(ctx_features)], dim=-1)
            pred_log_label = model.classifier_head(fused_for_log).argmax().item()
            fused_for_ctx = torch.cat([torch.zeros_like(log_features), ctx_features], dim=-1)
            pred_ctx_label = model.classifier_head(fused_for_ctx).argmax().item()
            
            llm_verdict_str = call_debate_llm(
                log_content=raw_logs[i],
                log_agent_pred=CLASS_MAP.get(pred_log_label, "Unknown"),
                context_agent_pred=CLASS_MAP.get(pred_ctx_label, "Unknown"),
                reason=reason_for_debate,
                confidence=confidence
            )
            final_pred = reverse_class_map.get(llm_verdict_str, 0)
        else:
            # 如果置信度足够高，直接采纳模型结果
            final_pred = prediction
        
        final_preds.append(final_pred)

print(f"\n[Debate] Total cases debated by LLM: {debated_cases}/{len(test_ds)} ({debated_cases/len(test_ds):.2%})")


# ================== 4. 评估最终结果 (不变) ==================
print("\n=== Final Report after Adjudication with Threshold ===")
print(f"Test F1(macro): {f1_score(y_true, final_preds, average='macro'):.6f}")
print(f"Test Acc:       {accuracy_score(y_true, final_preds):.6f}")
print("Per-class report:\n", classification_report(y_true, final_preds, digits=4, target_names=CLASS_MAP.values()))