import chromadb
import uuid
import requests


def ollama_embedding_by_api(text):
    res = requests.post(
        url="http://127.0.0.1:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    embedding = res.json()['embedding']
    return embedding


client = chromadb.PersistentClient(path="db/chroma_demo")  # 数据库 类似于=文件夹
collection = client.get_or_create_collection(name="collection_v1")  # 集合   类似于=表格

documents = ["风寒感冒", "寒邪客胃", "心脾两虚"]
ids = [str(uuid.uuid4()) for _ in documents]
embeddings = [ollama_embedding_by_api(text) for text in documents]

# 插入数据
collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

qs = "大小便失禁, 智能减退, 痴呆, 记忆力障碍"
qs_embedding = ollama_embedding_by_api(qs)

res = collection.query(query_embeddings=[qs_embedding, ],query_texts=qs, n_results=2)
print(res)