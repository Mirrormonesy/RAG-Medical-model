import uuid
import chromadb
import requests
import numpy as np
import re

def file_chunk_list():
    with open("/Users/mirror/Desktop/code/CODE/medical_filtered_60000.txt", encoding='utf-8') as fp:   
        data = fp.read()
    entries = data.split("名称:")
    entries = [e.strip() for e in entries if e.strip()]
    chunk_list = []
    for entry in entries:
        lines = entry.split("\n")
        name = lines[0].strip() if lines else ""
        entry_text = "\n".join(lines[1:]) if len(lines) > 1 else ""
        idx = entry_text.find("症状:")
        if idx != -1:
            content = entry_text[idx+len("症状:"):].strip()
            if len(content) > 0:
                chunk = f"名称:{name} 症状:{content}"
                chunk_list.append(chunk)
    print(f"最终chunks数量: {len(chunk_list)}")
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
            "model": "shaw/dmeta-embedding-zh:latest",
            "prompt": text
        }
    )
    embedding = res.json()['embedding']
    return l2_normalize(embedding)


def ollama_generate_by_api(prompt, system):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    payload = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }
    headers = {
        "Authorization": "Bearer sk-xaaxeehzqfhzfsdzsjdbwsujcgewigtpvyhjewztcrjhnqac",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        print("生成api解析失败：", e)
        return ""

def initial():
    client = chromadb.PersistentClient(path="db/chroma_demo")

    # 创建集合
    client.delete_collection("collection_v2")
    collection = client.get_or_create_collection(name="collection_v2")

    # 构造数据
    documents = file_chunk_list()
    ids = [str(uuid.uuid4()) for _ in range(len(documents))]
    embeddings = [ollama_embedding_by_api(text) for text in documents]

    # 分批插入数据
    batch_size = 98
    for i in range(0, len(documents), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size]
        )


def build_fullinfo_dict():
    with open("/Users/mirror/Desktop/code/CODE/medical_filtered_60000.txt", encoding='utf-8') as fp:
        data = fp.read()
    entries = [e.strip() for e in data.split("名称:") if e.strip()]
    info_dict = {}
    for entry in entries:
        lines = entry.split("\n")
        name = lines[0].strip() if lines else ""
        info_dict[name] = entry  # entry包含所有字段原文
    return info_dict


def run():
    # 关键字搜索
    qs = "内痔出血的特点是什么？"
    qs_embedding = ollama_embedding_by_api(qs)

    client = chromadb.PersistentClient(path="db/chroma_demo")
    collection = client.get_collection(name="collection_v2")
    res = collection.query(query_embeddings=[qs_embedding, ], query_texts=qs, n_results=3)
    print("距离：", res.get("distances"))
    docs = res["documents"][0]
    dists = res["distances"][0]
    disease_names = []
    for i, doc in enumerate(docs):
        print(f"TOP{i+1} 距离: {dists[i]:.4f}\n内容片段：{doc[:200]}\n")
        match = re.match(r"名称:([^\s]+)", doc)
        if match:
            disease_names.append(match.group(1))
    # 召回完整信息
    info_dict = build_fullinfo_dict()
    full_infos = [info_dict[name] for name in disease_names if name in info_dict]
    context = "\n\n".join(full_infos)
    prompt = r"""
我们从知识库中获取参考信息如下：
{context}
用户问题：
{qs}
""".format(context=context, qs=qs)
    system = r"""
你是一个医学问答机器人，任务是根据参考信息回答用户问题，
同时要求保证输出参考文献的80%，自行润色20%，
如果参考信息不足以回答用户问题，请回复不知道，
不要去杜撰任何信息。
请用英文回答。"""
    result = ollama_generate_by_api(prompt,system)
    print(result)


if __name__ == '__main__':
    initial()
    run()
