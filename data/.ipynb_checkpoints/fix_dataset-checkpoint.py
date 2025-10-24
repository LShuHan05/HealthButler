# fix_dataset.py
import json
import os

# 检查当前目录并确定正确的文件路径
current_dir = os.getcwd()
print(f"当前目录: {current_dir}")

# 尝试不同的路径
possible_paths = [
    'dirty_chinese_dpo.json',  # 如果在 data 目录中
    '../data/dirty_chinese_dpo.json',  # 如果在项目根目录
    'dirty_chinese_dpo.json'  # 直接在当前目录
]

data = None
for path in possible_paths:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功从 {path} 加载数据")
        break
    except FileNotFoundError:
        continue

if data is None:
    print("❌ 错误: 找不到数据文件")
    print("请确保数据文件存在于以下位置之一:")
    for path in possible_paths:
        print(f"  - {path}")
    exit()

print(f"原始数据条数: {len(data)}")

# 创建修复后的数据
fixed_data = []
for i, item in enumerate(data):
    fixed_item = {}
    
    # 处理 chosen - 确保是字符串
    chosen = item['chosen']
    if isinstance(chosen, dict) and 'value' in chosen:
        fixed_item['chosen'] = chosen['value']
    else:
        fixed_item['chosen'] = str(chosen)
    
    # 处理 rejected - 确保是字符串
    rejected = item['rejected']
    if isinstance(rejected, dict) and 'value' in rejected:
        fixed_item['rejected'] = rejected['value']
    else:
        fixed_item['rejected'] = str(rejected)
    
    # 处理 input - 从 conversations 中提取 human 输入
    human_input = ""
    if 'conversations' in item and item['conversations']:
        for turn in item['conversations']:
            if turn.get('from') == 'human' and 'value' in turn:
                human_input = turn['value']
                break
    
    fixed_item['input'] = human_input
    
    # 只保留有效数据
    if (fixed_item['input'] and 
        fixed_item['chosen'].strip() and 
        fixed_item['rejected'].strip()):
        fixed_data.append(fixed_item)

# 保存修复后的数据
output_path = 'dirty_chinese_rm_fixed.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(fixed_data, f, ensure_ascii=False, indent=2)

print(f"修复后数据条数: {len(fixed_data)}")
print(f"修复后的数据已保存到: {output_path}")

# 验证修复结果
if fixed_data:
    sample = fixed_data[0]
    print(f"\n示例数据验证:")
    print(f"chosen 类型: {type(sample['chosen'])}, 内容: {repr(sample['chosen'][:50])}...")
    print(f"rejected 类型: {type(sample['rejected'])}, 内容: {repr(sample['rejected'][:50])}...")
    print(f"input 类型: {type(sample['input'])}, 内容: {repr(sample['input'][:50])}...")