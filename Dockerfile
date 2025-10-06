# Використовуємо офіційний, легкий образ Python 3.11
FROM python:3.12-slim

# Встановлюємо робочу директорію всередині контейнера
WORKDIR /app

# Копіюємо файл з бібліотеками
COPY ../requirements.txt .

# Встановлюємо бібліотеки
RUN pip install --no-cache-dir -r requirements.txt

# Завантажуємо необхідний словник для NLTK
RUN python -m nltk.downloader vader_lexicon

# Копіюємо решту коду вашого додатку
COPY .. .

# Вказуємо команду, яку потрібно виконати при запуску контейнера
CMD ["python", "main.py"]