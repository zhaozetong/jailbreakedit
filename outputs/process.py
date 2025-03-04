import json


# 读取未格式化的 JSON 文件
with open('llama-7b-1-delta_name--llama-7b-misuse.json', 'r', encoding='utf-8') as f:
    data = json.load(f)  # 加载 JSON 数据

# 将 JSON 数据格式化并保存到新文件
with open('what.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)  # 使用缩进和换行符