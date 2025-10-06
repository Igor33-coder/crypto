# Використовуємо Python 3.12
FROM python:3.12-slim

WORKDIR /app

# --- ▼▼▼ ОНОВЛЕНИЙ БЛОК ▼▼▼ ---
# Встановлюємо системні залежності, необхідні для TA-Lib
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Завантажуємо та компілюємо TA-Lib зі джерела
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install

# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

COPY requirements.txt .

# Тепер pip install зможе знайти скомпільовану TA-Lib
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader vader_lexicon

COPY . .

CMD ["python", "main.py"]