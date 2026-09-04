FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY scripts/ ./scripts/
COPY knowledge_base/ ./knowledge_base/
COPY faiss_index/ ./faiss_index/

# Переменные окружения
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUNBUFFERED=1

# Запуск RAG-бота
CMD ["python", "scripts/rag_bot.py"]
