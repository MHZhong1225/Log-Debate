# ===============================================================
# preprocess_all.py
# ===============================================================
"""
统一预处理四个日志数据集：
  1. Hadoop（硬编码 AppId→4 类标签，行级 + train 子集分层抽样）
  2. HDFS  （BlockId→Normal/Anomaly，行级+窗口）
  3. BGL   （-" 前缀标记是否异常，行级+窗口）
  4. Tbird （"-" 前缀标记是否异常，行级）
---------------------------------------------------------------
运行示例
  python preprocess_all.py hadoop   # 只跑 Hadoop
  python preprocess_all.py all      # 依次跑完四个
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Union
import argparse, re, os, sys
import pandas as pd
import numpy as np

# ================ 公共工具 ================
def ensure_out(out: Path):
    out.mkdir(parents=True, exist_ok=True)

def ratio_split(df: pd.DataFrame, ratios: Dict[str, float]) -> Dict[str, pd.DataFrame]:
    assert abs(sum(ratios.values()) - 1) < 1e-6
    n, splits, cur = len(df), {}, 0
    for k, r in ratios.items():
        sz = int(round(n * r))
        splits[k] = df.iloc[cur:] if k == list(ratios)[-1] else df.iloc[cur:cur + sz]
        splits[k] = splits[k].reset_index(drop=True); cur += sz
    return splits

def save_splits(splits: Dict[str, pd.DataFrame], prefix: str, out: Path):
    ensure_out(out)
    for name, df in splits.items():
        fn = out / f"{prefix}_{name}.csv"
        df.to_csv(fn, index=False)
        print(f"[✓] {fn.name:20s}: {len(df):,} 行 {dict(df.label.value_counts().sort_index())}")

def stratified_head_fraction(df: pd.DataFrame, frac: float,
                             min_per_class: int = 1) -> pd.DataFrame:
    """
    在保持类别比例的前提下，按时间顺序（原index顺序）从每个类别取前 k 条，
    其中 k = max(min_per_class, round(n_label * frac))。
    返回保持原始相对顺序的拼接结果。
    """
    picked_idx = []
    for lab, sub in df.groupby("label", sort=False):
        n_lab = len(sub)
        k = max(min_per_class, int(round(n_lab * frac)))
        picked_idx.extend(sub.index[:k])
    picked_idx.sort()
    return df.loc[picked_idx].reset_index(drop=True)

# ================ 1. Hadoop 行级预处理 ================
def preprocess_hadoop():
    print("\n==== Hadoop ====")
    # --- AppId→label ---
    # WordCount
    # NORMAL = {
    #     "application_1445087491445_0005",
    #     "application_1445087491445_0007",
    #     "application_1445175094696_0005",
    # }
    # MACHINE_DOWN = {
    #     "application_1445087491445_0001", "application_1445087491445_0002",
    #     "application_1445087491445_0003", "application_1445087491445_0004",
    #     "application_1445087491445_0006", "application_1445087491445_0008",
    #     "application_1445087491445_0009", "application_1445087491445_0010",
    #     "application_1445094324383_0001", "application_1445094324383_0002",
    #     "application_1445094324383_0003", "application_1445094324383_0004",
    #     "application_1445094324383_0005",
    # }
    # NETWORK_DISC = {
    #     "application_1445175094696_0001", "application_1445175094696_0002",
    #     "application_1445175094696_0003", "application_1445175094696_0004",
    # }
    # DISK_FULL = {
    #     "application_1445182159119_0001", "application_1445182159119_0002",
    #     "application_1445182159119_0003", "application_1445182159119_0004",
    #     "application_1445182159119_0005",
    # }

    # PageRank
    NORMAL = {
        "application_1445062781478_0011", "application_1445144423722_0021",
        "application_1445062781478_0016", "application_1445076437777_0005",
        "application_1445062781478_0019","application_1445076437777_0002",
        "application_1445144423722_0024", "application_1445182159119_0012",
    }
    MACHINE_DOWN = {
        "application_1445062781478_0012", "application_1445062781478_0013",
        "application_1445062781478_0014", "application_1445062781478_0015",
        "application_1445062781478_0017", "application_1445062781478_0018",
        "application_1445062781478_0020", "application_1445076437777_0001",
        "application_1445076437777_0003", "application_1445076437777_0004",
        "application_1445182159119_0016", "application_1445182159119_0017",
        "application_1445182159119_0018", "application_1445182159119_0019","application_1445182159119_0020"
    }
    NETWORK_DISC = {
        "application_1445144423722_0020", "application_1445144423722_0022",
        "application_1445144423722_0023",
    }
    DISK_FULL = {
        "application_1445182159119_0011", "application_1445182159119_0013",
        "application_1445182159119_0014", "application_1445182159119_0015",
    }

    LABEL2INT = {
        "normal": 0,
        "machine_down": 1,
        "network_disconnection": 2,
        "disk_full": 3,
    }
    # 反向查找字典：AppId → label_int
    APP2LABEL: Dict[str, int] = {}
    for app in NORMAL:        APP2LABEL[app] = LABEL2INT["normal"]
    for app in MACHINE_DOWN:  APP2LABEL[app] = LABEL2INT["machine_down"]
    for app in NETWORK_DISC:  APP2LABEL[app] = LABEL2INT["network_disconnection"]
    for app in DISK_FULL:     APP2LABEL[app] = LABEL2INT["disk_full"]

    # ========= 2. 基础配置（可自行修改） =========
    CFG = dict(
        input_log="./datasets/Hadoop/raw",     # 日志目录或单文件
        output_dir="./datasets/Hadoop/",       # 输出目录
        window_size=100,
        step_size=100,
        ratios={"train": 0.8, "val": 0.1, "test": 0.1},
        small_ratios={"0p1": 0.001, "0p5": 0.005, "1p0": 0.01},
        separator=" ",
    )

    aid_pat = re.compile(r"application_\d+_\d+")

    def read_lines(root: Path):
        for fp in sorted(p for p in root.rglob("*") if p.is_file()):
            m = aid_pat.search(fp.as_posix())
            fallback = APP2LABEL.get(m.group(0) if m else None, 0)
            with fp.open(encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    ln = ln.rstrip("\n");  ids = aid_pat.findall(ln)
                    lab = max(APP2LABEL.get(i, fallback) for i in ids) if ids else fallback
                    yield lab, ln
    def stratified_time_split(df: pd.DataFrame,
                            ratios: Dict[str, float]) -> Dict[str, pd.DataFrame]:
        assert abs(sum(ratios.values())-1) < 1e-6
        splits_idx = {k: [] for k in ratios}
        # 对每个 label 独立切，再合并回全局时间顺序
        for lab, sub in df.groupby("label", sort=False):
            n = len(sub)
            n_train = int(round(n*ratios["train"]))
            n_val   = int(round(n*ratios["val"]))
            idx = sub.index
            splits_idx["train"].extend(idx[:n_train])
            splits_idx["val"].extend(idx[n_train:n_train+n_val])
            splits_idx["test"].extend(idx[n_train+n_val:])
        return {k: df.loc[sorted(v)].reset_index(drop=True) for k, v in splits_idx.items()}
    def stratified_subsample(df: pd.DataFrame, frac: float,
                            min_per_class: int = 1) -> pd.DataFrame:
        """
        对行级 df 按 label 分层抽样，整体采样率 = frac。
        确保每个 label 至少 min_per_class 行。
        按原始顺序（index）保持时间一致。
        """
        picked_idx = []
        for lab, sub in df.groupby("label", sort=False):
            n_lab = len(sub)
            k = max(min_per_class, int(round(n_lab * frac)))
            picked_idx.extend(sub.index[:k])          # 取最早 k 行即可
        picked_idx.sort()
        return df.loc[picked_idx].reset_index(drop=True)
    labels, texts = zip(*read_lines(Path(CFG["input_log"])))

    df = pd.DataFrame(dict(label=labels, content=texts))
    df = df.dropna(subset=['content'])    
    df = df[ df['content'].str.strip().astype(bool) ]
    print("总行数 / label 分布:", len(df), dict(df.label.value_counts().sort_index()))

    print("[*] 分层时间切分…")
    split = stratified_time_split(df, CFG["ratios"])
    # save_splits({f"log_{k}": v for k,v in split.items()}, Path(CFG["output_dir"]))
    save_splits(split, "log", Path(CFG["output_dir"]))
    print("[*] 采样小比例 train 子集…")
    train_df = split["train"]
    for tag, frac in CFG["small_ratios"].items():
        sub = stratified_subsample(train_df, frac, min_per_class=1)
        sub.to_csv(Path(CFG["output_dir"]) / f"log_train_{tag}.csv", index=False)
        print(f"  [✓] log_train_{tag}: {len(sub)} 行（label 分布：{dict(sub.label.value_counts().sort_index())}）")

# ================ 2. HDFS (BlockId) ================
def preprocess_hdfs():
    print("\n==== HDFS ====")
    INPUT_LOG  = Path("./datasets/HDFS/HDFS.log")
    LABEL_CSV  = Path("./datasets/HDFS/anomaly_label.csv")
    OUT        = Path("./datasets/HDFS")
    win, step  = 100, 100
    ratios     = {"train":.8,"val":.1,"test":.1}

    blk_map = {str(bid): 0 if lab.lower()=="normal" else 1
               for bid, lab in pd.read_csv(LABEL_CSV).values}
    pat = re.compile(r"blk_[\-\d]+")

    labels, texts = [], []
    with INPUT_LOG.open(encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ids = pat.findall(ln); lab = max(blk_map.get(i,0) for i in ids) if ids else 0
            labels.append(lab); texts.append(ln.rstrip("\n"))
    df_line = pd.DataFrame(dict(label=labels, content=texts))
    print("总行数 / label 分布:", len(df_line), dict(df_line.label.value_counts().sort_index()))
    df_line = df_line.dropna(subset=['content'])   # ← NEW

    # 简单窗口
    n = len(df_line); win_id = np.full(n,-1,int); w=0
    for s in range(0,n,step): win_id[s:min(s+win,n)]=w; w+=1
    df_line["win_id"]=win_id
    df_win = df_line.groupby("win_id").agg(label=('label','max'),
                                           content=('content',' '.join)).reset_index()

    split_win = ratio_split(df_win, ratios)
    # 行级映射
    split_log = {k: df_line[df_line.win_id.isin(set(v.win_id))].reset_index(drop=True)
                 for k,v in split_win.items()}
    save_splits(split_log, "log", OUT)

# ================ 3. BGL ================
def preprocess_bgl():
    print("\n==== BGL ====")
    INPUT = Path("./datasets/BGL/BGL.log")
    OUT   = Path("./datasets/BGL")
    win, step = 100, 100
    ratios = {"train":.8,"val":.1,"test":.1}

    labels, texts = [], []
    with INPUT.open(encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.rstrip("\n"); first, rest = ln.split(maxsplit=1)
            labels.append(0 if first=="-" else 1); texts.append(rest)
    df_line = pd.DataFrame(dict(label=labels, content=texts))
    print("总行数 / label 分布:", len(df_line), dict(df_line.label.value_counts().sort_index()))
    df_line = df_line.dropna(subset=['content'])   # ← NEW
    # 滑动窗口
    n=len(df_line); win_id=np.full(n,-1,int); w=0
    for s in range(0,n,step): win_id[s:min(s+win,n)]=w; w+=1
    df_line["win_id"]=win_id

    df_win = df_line.groupby("win_id").agg(label=('label','max'),
                                           content=('content',' '.join)).reset_index()
    split_win = ratio_split(df_win, ratios)
    split_log = {k: df_line[df_line.win_id.isin(set(v.win_id))].reset_index(drop=True)
                 for k,v in split_win.items()}
    save_splits(split_log, "log", OUT)

# ================ 4. Tbird ================
def preprocess_tbird():
    print("\n==== Tbird ====")
    INPUT = Path("./datasets/Tbird/Tbird.log")
    OUT   = Path("./datasets/Tbird")
    ratios = {"train":.8,"val":.1,"test":.1}
    small  = {"0p1":.001,"0p5":.005,"1p0":.01}

    rows=[]
    with INPUT.open(encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln=ln.rstrip("\\n"); first,*rest=ln.split(maxsplit=1)
            rows.append((0 if first=="-" else 1, rest[0] if rest else ""))
    df = pd.DataFrame(rows, columns=["label","content"])
    df = df.dropna(subset=['content'])             # ← NEW
    print("总行数 / label 分布:", len(df), dict(df.label.value_counts().sort_index()))

    split = ratio_split(df, ratios)
    save_splits(split, "log", OUT)

    # 小比例子集（分层抽样，保持类别比例；按时间顺序取各类前 k%）
    for tag, frac in small.items():
        for split_name in ["train","val","test"]:
            sub = stratified_head_fraction(split[split_name], frac, min_per_class=1)
            out_fp = OUT / f"log_{split_name}_{tag}.csv"
            sub.to_csv(out_fp, index=False)
            dist = dict(sub.label.value_counts().sort_index())
            print(f"[✓] {out_fp.name} : {len(sub)} 行 {dist}")

# ================ 入口 ================
DATASET_FUNCS = {"hadoop": preprocess_hadoop,
                 "hdfs":   preprocess_hdfs,
                 "bgl":    preprocess_bgl,
                 "tbird":  preprocess_tbird}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理四个公开日志数据集")
    parser.add_argument("dataset", nargs="?", default="all",
                        choices=list(DATASET_FUNCS)+["all"],
                        help="要处理的数据集名称 (hadoop/hdfs/bgl/tbird/all)")
    args = parser.parse_args()

    if args.dataset == "all":
        for name, fn in DATASET_FUNCS.items(): fn()
    else:
        DATASET_FUNCS[args.dataset]()

# python prepress.py hdfs
# python prepress.py hadoop