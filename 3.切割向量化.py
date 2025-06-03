import requests
import functools
import ollama
import numpy as np


def file_chunk_list():
    # 1.读取文件内容（不再只取前500行，取全部内容）
    with open("medical_filtered_60000.txt", encoding='utf-8', mode='r') as fp:
        data = fp.read()
    # 2.按“名称:”切割为疾病条目
    entries = data.split("名称:")
    entries = [e.strip() for e in entries if e.strip()]
    # 3.每个条目按字段切割，生成更小的片段
    fields = ["症状:", "诊断:", "治疗:", "用药:", "预防:", "病因:", "检查:"]
    chunk_list = []
    for entry in entries:
        lines = entry.split("\n")
        name = lines[0].strip() if lines else ""
        entry_text = "\n".join(lines[1:]) if len(lines) > 1 else ""
        field_indices = []
        for field in fields:
            idx = entry_text.find(field)
            if idx != -1:
                field_indices.append((idx, field))
        field_indices.sort()
        for j, (start_idx, field) in enumerate(field_indices):
            end_idx = field_indices[j+1][0] if j+1 < len(field_indices) else len(entry_text)
            content = entry_text[start_idx+len(field):end_idx].strip()
            if content:
                chunk_list.append(f"名称:{name} {field}{content}")
    return chunk_list


def l2_normalize(vec):
    arr = np.array(vec)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


def ollama_embedding_by_api(text):
    res = requests.post(
        url="http://127.0.0.1:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    embedding = res.json()['embedding']
    return l2_normalize(embedding)


def run():
    chunk_list = file_chunk_list()
    for i, chunk in enumerate(chunk_list[:20]):
        vector = ollama_embedding_by_api(chunk)
        print(vector)



if __name__ == '__main__':
    run()