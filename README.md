# RAG-бот QuantumForge Software

RAG-бот (Retrieval-Augmented Generation) для поиска по базе знаний компании QuantumForge Software.

## Описание

Бот помогает сотрудникам быстро находить информацию в корпоративной базе знаний, отвечая на вопросы на основе релевантных документов. Использует векторный поиск (FAISS) и LLM (YandexGPT) для формирования точных ответов.

## Технологии

| Компонент | Технология |
|-----------|------------|
| **Эмбеддинги** | all-MiniLM-L6-v2 (384d, локально) |
| **Векторная БД** | FAISS (IndexFlatIP) |
| **LLM** | YandexGPT Lite (API) |
| **Текстовые сплиттеры** | LangChain Text Splitters |
| **Язык** | Python 3.11+ |

## Структура проекта

```
├── knowledge_base/       # База знаний (33 документа)
├── faiss_index/          # Векторный индекс
├── scripts/
│   ├── build_index.py    # Построение индекса
│   ├── rag_bot.py        # RAG-бот (REPL)
│   └── test_bot.py       # Тестирование (10 запросов)
├── terms_map.json        # Словарь замен терминов
├── Project_template.md   # Отчёт по заданиям
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Запуск

### 1. Установка зависимостей

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Построение индекса

```bash
python scripts/build_index.py
```

### 3. Запуск бота

```bash
# С YandexGPT (требуется API ключ)
export YANDEX_IAM_TOKEN="your-token"
export YANDEX_FOLDER_ID="your-folder-id"
python scripts/rag_bot.py

# Без API (демо-режим)
python scripts/rag_bot.py
```

### 4. Docker

```bash
docker-compose up --build
```

## Тестирование

```bash
python scripts/test_bot.py
```

10 тестовых запросов: 5 успешных ответов + 5 отказов/фильтров.

## Лицензия

Проект для educational purposes (Яндекс Практикум, спринт 7).
