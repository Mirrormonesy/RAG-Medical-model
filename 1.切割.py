# 1.读取整个文件内容
with open("/Users/mirror/Desktop/code/REAL CODE/medical_filtered_60000.txt", encoding='utf-8', mode='r') as fp:
    data = fp.read()

# 2.按“名称:”切割为疾病条目
entries = data.split("名称:")
entries = [e.strip() for e in entries if e.strip()]

# 3.每个条目只保留症状字段，合并为一个chunk
fields = ["症状:"]
chunk_list = []
for entry in entries:
    lines = entry.split("\n")
    name = lines[0].strip() if lines else ""
    entry_text = "\n".join(lines[1:]) if len(lines) > 1 else ""
    idx = entry_text.find("症状:")
    if idx != -1:
        # 找到下一个字段的起始位置（此处只有一个字段，直接取到结尾）
        content = entry_text[idx+len("症状:"):].strip()
        # 只保留有内容的chunk
        if len(content) > 0:
            chunk = f"名称:{name} 症状:{content}"
            chunk_list.append(chunk)

print(f"总共切割出 {len(chunk_list)} 个chunk")
with open("chunk_output.txt", "w", encoding="utf-8") as f:
    for chunk in chunk_list:
        f.write(chunk + "\n\n")
