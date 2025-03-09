import json
import numpy as np
import faiss
import requests
from docx import Document

# Загрузка документа и разбиение на чанки
def load_document(file_path):
    doc = Document(file_path)
    text = " ".join([paragraph.text for paragraph in doc.paragraphs])
    return text

def split_into_chunks(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Создание эмбеддингов через Ollama API
def get_embedding(text, model="snowflake-arctic-embed2:latest"):
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": model,
        "prompt": text
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        raise Exception(f"Ошибка: {response.status_code}, {response.text}")

# Сохранение эмбеддингов в JSON
def save_embeddings_to_json(embeddings, chunks, file_path="embeddings.json"):
    data = {
        "embeddings": [embedding.tolist() for embedding in embeddings],
        "chunks": chunks
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Эмбеддинги сохранены в {file_path}")

# Загрузка эмбеддингов из JSON
def load_embeddings_from_json(file_path="embeddings.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    embeddings = np.array(data["embeddings"])
    chunks = data["chunks"]
    print(f"Эмбеддинги загружены из {file_path}")
    return embeddings, chunks

# Поиск по индексу
def search(query, top_k=3):
    query_embedding = get_embedding(query)
    query_embedding = np.array([query_embedding])
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for idx in indices[0]:
        results.append(chunks[idx])
    
    return results

# Генерация ответа через Ollama API
def generate_response(query, context):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:latest",
        "prompt": f"Запрос: {query}\nКонтекст: {context}\nОтвет:",
        "stream": False
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Ошибка: {response.status_code}"

# Основной пайплайн
file_path = "for_test.docx"
text = load_document(file_path)
chunks = split_into_chunks(text)

# Проверяем, есть ли сохраненные эмбеддинги
try:
    embeddings, chunks = load_embeddings_from_json()
except FileNotFoundError:
    # Если файла нет, создаем эмбеддинги и сохраняем их
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
    embeddings = np.array(embeddings)
    save_embeddings_to_json(embeddings, chunks)

# Создание FAISS индекса
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Поиск по запросу
query = "В какой секте учился Хань Ли"
results = search(query)

# Генерация ответа
context = "\n".join(results)
response = generate_response(query, context)

print("Сгенерированный ответ:")
print(response)
