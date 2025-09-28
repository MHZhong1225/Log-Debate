import os
import pandas as pd
from tqdm import tqdm

# --- 步骤1: 标签映射 (不变) ---
label_mapping = {
    'application_1445087491445_0005': 'Normal',
    'application_1445087491445_0007': 'Normal',
    'application_1445175094696_0005': 'Normal',
    'application_1445087491445_0001': 'Machine down',
    'application_1445087491445_0002': 'Machine down',
    'application_1445087491445_0003': 'Machine down',
    'application_1445087491445_0004': 'Machine down',
    'application_1445087491445_0006': 'Machine down',
    'application_1445087491445_0008': 'Machine down',
    'application_1445087491445_0009': 'Machine down',
    'application_1445087491445_0010': 'Machine down',
    'application_1445094324383_0001': 'Machine down',
    'application_1445094324383_0002': 'Machine down',
    'application_1445094324383_0003': 'Machine down',
    'application_1445094324383_0004': 'Machine down',
    'application_1445094324383_0005': 'Machine down',
    'application_1445175094696_0001': 'Network disconnection',
    'application_1445175094696_0002': 'Network disconnection',
    'application_1445175094696_0003': 'Network disconnection',
    'application_1445175094696_0004': 'Network disconnection',
    'application_1445182159119_0001': 'Disk full',
    'application_1445182159119_0002': 'Disk full',
    'application_1445182159119_0003': 'Disk full',
    'application_1445182159119_0004': 'Disk full',
    'application_1445182159119_0005': 'Disk full',
    'application_1445062781478_0011': 'Normal',
    'application_1445062781478_0016': 'Normal',
    'application_1445062781478_0019': 'Normal',
    'application_1445076437777_0002': 'Normal',
    'application_1445076437777_0005': 'Normal',
    'application_1445144423722_0021': 'Normal',
    'application_1445144423722_0024': 'Normal',
    'application_1445182159119_0012': 'Normal',
    'application_1445062781478_0012': 'Machine down',
    'application_1445062781478_0013': 'Machine down',
    'application_1445062781478_0014': 'Machine down',
    'application_1445062781478_0015': 'Machine down',
    'application_1445062781478_0017': 'Machine down',
    'application_1445062781478_0018': 'Machine down',
    'application_1445062781478_0020': 'Machine down',
    'application_1445076437777_0001': 'Machine down',
    'application_1445076437777_0003': 'Machine down',
    'application_1445076437777_0004': 'Machine down',
    'application_1445182159119_0016': 'Machine down',
    'application_1445182159119_0017': 'Machine down',
    'application_1445182159119_0018': 'Machine down',
    'application_1445182159119_0019': 'Machine down',
    'application_1445182159119_0020': 'Machine down',
    'application_1445144423722_0020': 'Network disconnection',
    'application_1445144423722_0022': 'Network disconnection',
    'application_1445144423722_0023': 'Network disconnection',
    'application_1445182159119_0011': 'Disk full',
    'application_1445182159119_0013': 'Disk full',
    'application_1445182159119_0014': 'Disk full',
    'application_1445182159119_0015': 'Disk full',
}

# --- 步骤2: 读取日志内容并生成数据---
BASE_LOG_DIRECTORY = './datasets/Hadoop/raw/' # 请确保此路径正确，推荐使用绝对路径

all_log_records = []
log_line_counter = 0

# 检查路径是否存在
if not os.path.isdir(BASE_LOG_DIRECTORY):
    print(f"错误：找不到目录 '{BASE_LOG_DIRECTORY}'。请检查路径是否正确。")
    exit()

# 获取所有 application ID (文件夹名)
app_folders = [d for d in os.listdir(BASE_LOG_DIRECTORY) 
               if d.startswith('application_') and os.path.isdir(os.path.join(BASE_LOG_DIRECTORY, d))]

print(f"发现 {len(app_folders)} 个 application 文件夹...")

for app_id in tqdm(app_folders, desc="处理文件夹"):
    label = label_mapping.get(app_id, 'Unknown')
    app_folder_path = os.path.join(BASE_LOG_DIRECTORY, app_id)
    
    for root, dirs, files in os.walk(app_folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        content = line.strip()
                        if content:
                            record = {
                                'id': log_line_counter,  # <--- 修改2: 使用计数器作为ID
                                'label': label,
                                'content': content,
                                # 'source_application_id': app_id # 新增: 保留原始ID
                            }
                            all_log_records.append(record)
                            log_line_counter += 1  # <--- 修改3: 计数器加一
            except Exception as e:
                print(f"无法读取文件 {file_path}: {e}")

print("\n日志内容读取完毕，开始生成CSV文件...")

# --- 步骤3: 保存为CSV文件 (不变) ---
if all_log_records:
    df = pd.DataFrame(all_log_records)
    output_filename = 'hadoop_logs_full.csv'
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n处理完成！数据已成功保存到 {output_filename}")
    print(f"总共生成了 {len(df)} 条日志记录。")
    print("\n文件预览 (前5行):")
    print(df.head())
else:
    print("\n处理完成，但没有找到任何日志记录。请检查文件夹是否为空。")