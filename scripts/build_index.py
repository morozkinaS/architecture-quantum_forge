"""
build_index.py — Задание 3: Создание векторного индекса базы знаний.

Загружает документы из knowledge_base/, разбивает на чанки,
генерирует эмбеддинги (all-MiniLM-L6-v2) и сохраняет индекс FAISS.
"""

import os
import json
import time
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Конфигурация ──────────────────────────────────────────────
KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge_base"
INDEX_DIR = Path(__file__).parent.parent / "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500       # символов
CHUNK_OVERLAP = 100    # символов


def load_documents(data_dir: Path) -> list[dict]:
    """Загружает все .txt файлы из директории."""
    docs = []
    for file_path in sorted(data_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({
                "text": text,
                "metadata": {
                    "source": file_path.name,
                    "title": file_path.stem.replace("_", " ").title(),
                },
            })
    print(f"  Загружено документов: {len(docs)}")
    return docs


def split_documents(
    docs: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Разбивает документы на чанки с сохранением метаданных."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_id": len(chunks),
                    "chunk_index": i,
                },
            })

    print(f"  Всего чанков: {len(chunks)} (средний размер: "
          f"{sum(len(c['text']) for c in chunks) // max(len(chunks), 1)} символов)")
    return chunks


def build_faiss_index(
    chunks: list[dict],
    model_name: str = EMBEDDING_MODEL,
) -> tuple[faiss.Index, np.ndarray]:
    """Создаёт FAISS-индекс из чанков."""
    print(f"  Загрузка модели эмбеддингов: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"  Генерация эмбеддингов для {len(texts)} чанков...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")

    # Нормализация для cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine after normalization
    index.add(embeddings)

    print(f"  Индекс создан: {index.ntotal} векторов, размерность {dimension}")
    return index, embeddings


def save_index(
    index: faiss.Index,
    chunks: list[dict],
    index_dir: Path,
):
    """Сохраняет индекс и метаданные на диск."""
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "faiss.index"))

    with open(index_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    # Сохраняем конфигурацию
    config = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "total_chunks": len(chunks),
        "total_documents": len(set(c["metadata"]["source"] for c in chunks)),
        "dimension": index.d,
    }
    with open(index_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  Индекс сохранён в {index_dir}/")


def test_index(index: faiss.Index, chunks: list[dict], model_name: str = EMBEDDING_MODEL):
    """Проверяет качество индекса тестовыми запросами."""
    model = SentenceTransformer(model_name)

    test_queries = [
        "Кто такой Xarn Velgor?",
        "Что такое Void Core?",
        "Как устроен flux drive?",
        "Кто такое Order of Vexari?",
        "Что happened on Pyrexis?",
    ]

    print("\n  ── Тестовые запросы ──")
    for query in test_queries:
        q_emb = model.encode([query])
        faiss.normalize_L2(q_emb)
        distances, indices = index.search(q_emb, k=3)

        print(f"\n  Запрос: {query}")
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            source = chunks[idx]["metadata"]["source"]
            preview = chunks[idx]["text"][:100].replace("\n", " ")
            print(f"    {rank+1}. [{source}] (score={dist:.3f}) {preview}...")


def main():
    print("═══ Задание 3: Создание векторного индекса ═══\n")

    start = time.time()

    print("[1/5] Загрузка документов...")
    docs = load_documents(KNOWLEDGE_DIR)

    print("\n[2/5] Разбиение на чанки...")
    chunks = split_documents(docs)

    print("\n[3/5] Построение индекса FAISS...")
    index, embeddings = build_faiss_index(chunks)

    print("\n[4/5] Сохранение индекса...")
    save_index(index, chunks, INDEX_DIR)

    print("\n[5/5] Тестирование...")
    test_index(index, chunks)

    elapsed = time.time() - start
    print(f"\n═══ Готово! Время выполнения: {elapsed:.1f} сек ═══")


if __name__ == "__main__":
    main()
