# create_rm_dataset.py
import json
import os

# 获取当前目录
current_dir = os.getcwd()
print(f"当前目录: {current_dir}")

# 确定正确的文件路径
if current_dir.endswith('/data'):
    # 如果在 data 目录中
    input_file = 'dirty_chinese_dpo.json'
    output_file = 'dirty_chinese_rm_working.json'
else:
    # 如果在项目根目录
    input_file = 'data/dirty_chinese_dpo.json'
    output_file = 'data/dirty_chinese_rm_working.json'

print(f"输入文件: {input_file}")
print(f"输出文件: {output_file}")

# 读取原始数据
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功加载数据，共 {len(data)} 条")
except FileNotFoundError:
    print(f"❌错误: 找不到文件 {input_file}")
    print("请确保在正确的目录中运行脚本")
    exit()

# 创建修复后的数据
fixed_data = []
skipped_count = 0

for i, item in enumerate(data):
    try:
        # 提取 chosen
        chosen = item.get('chosen', '')
        if isinstance(chosen, dict):
            chosen = chosen.get('value', '')
        chosen = str(chosen).strip()
        
        # 提取 rejected
        rejected = item.get('rejected', '')
        if isinstance(rejected, dict):
            rejected = rejected.get('value', '')
        rejected = str(rejected).strip()
        
        # 提取 human input
        human_input = ""
        if 'conversations' in item:
            for turn in item['conversations']:
                if turn.get('from') == 'human':
                    # 处理 human 输入
                    human_value = turn.get('value', '')
                    if isinstance(human_value, str):
                        human_input = human_value.strip()
                    break
        
        # 验证数据
        if not human_input:
            skipped_count += 1
            continue
            
        if not chosen:
            skipped_count += 1
            continue
            
        if not rejected:
            skipped_count += 1
            continue
        
        fixed_data.append({
            "input": human_input,
            "chosen": chosen,
            "rejected": rejected
        })
        
    except Exception as e:
        skipped_count += 1
        continue

# 保存修复后的数据
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(fixed_data, f, ensure_ascii=False, indent=2)

print(f"\n=== 处理结果 ===")
print(f"原始数据: {len(data)} 条")
print(f"修复后数据: {len(fixed_data)} 条")
print(f"跳过数据: {skipped_count} 条")
print(f"保存到: {output_file}")

# 验证数据格式
if fixed_data:
    print(f"\n=== 数据格式验证 ===")
    sample = fixed_data[0]
    print(f"第一条数据:")
    print(f"  input: {repr(sample['input'][:50])}...")
    print(f"  chosen: {repr(sample['chosen'][:50])}...")
    print(f"  rejected: {repr(sample['rejected'][:50])}...")