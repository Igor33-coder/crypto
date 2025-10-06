import os
import asyncpraw as praw
import asyncio
import logging
import aiohttp
import numpy as np
import time
import hmac
import hashlib
import google.generativeai as genai
import json
from pybit.unified_trading import HTTP
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# --- ▼▼▼ НОВІ ІМПОРТИ ДЛЯ ШІ ТА НОВИН ▼▼▼ ---
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --------------------------
# Логування
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --------------------------
# API ключі та токени
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

BASE_URL = "https://api.binance.com"

# --- ▼▼▼ Додайте ці ключі разом з іншими ▼▼▼ ---
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = "CryptoSentimentBot/1.0" # Може бути будь-який унікальний рядок

# Ініціалізація Reddit клієнта (додайте це після sia = SentimentIntensityAnalyzer())
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
)

# --- ▼▼▼ Додайте ключ Gemini разом з іншими ▼▼▼ ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Налаштовуємо Gemini
genai.configure(api_key=GEMINI_API_KEY)

# --- НОВА, НАДІЙНА ІНІЦІАЛІЗАЦІЯ МОДЕЛІ ---
# Програмно отримуємо список доступних моделей і беремо першу, що підтримує 'generateContent'
try:
    available_models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    if not available_models:
        raise ValueError("Не знайдено жодної доступної моделі Gemini, що підтримує генерацію контенту.")

    # Вибираємо першу модель зі списку (зазвичай це 'gemini-pro' або аналог)
    model_name = available_models[0].name
    model = genai.GenerativeModel(model_name)
    logger.info(f"Успішно ініціалізовано модель Gemini: {model_name}")

except Exception as e:
    logger.error(f"Критична помилка ініціалізації моделі Gemini: {e}")
    # Якщо модель не ініціалізувалася, бот не зможе працювати з ШІ.
    # Можна або зупинити бота, або встановити 'model = None' і обробляти це в 'get_llm_analysis'
    model = None
# ----------------------------------------------------

# --------------------------
# Ініціалізація клієнтів
sia = SentimentIntensityAnalyzer()
# --------------------------

# --------------------------
# --- АРХІТЕКТУРА АДАПТЕРІВ ДЛЯ БІРЖ ---
# --------------------------

class ExchangeAdapter:
    """Базовий клас-шаблон для всіх бірж."""

    def __init__(self):
        self.name = "Unknown"
        self.session = None

    async def get_klines(self, symbol, interval, limit):
        """Метод для отримання 'свічок' (klines). Має бути реалізований кожним нащадком."""
        raise NotImplementedError("Метод get_klines має бути реалізований.")

    async def get_market_tickers(self, session):
        """Метод для отримання даних для сканера ринку."""
        raise NotImplementedError("Метод get_market_tickers має бути реалізований.")

    def format_klines_data(self, data, symbol):
        """Перетворює унікальні дані біржі у наш стандартний формат."""
        raise NotImplementedError("Метод format_klines_data має бути реалізований.")


class BinanceAdapter(ExchangeAdapter):
    """Адаптер для біржі Binance."""

    def __init__(self):
        super().__init__()
        self.name = "Binance"
        self.base_url = "https://api.binance.com"

    async def get_klines(self, session, symbol, interval='1h', limit=100):
        url = f"{self.base_url}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        async with session.get(url) as resp:
            data = await resp.json()
            return self.format_klines_data(data, symbol)

    async def get_market_tickers(self, session):
        url = f"{self.base_url}/api/v3/ticker/24hr"
        async with session.get(url) as resp:
            return await resp.json()

    def format_klines_data(self, data, symbol):
        if not isinstance(data, list) or not data:
            raise ValueError(f"Некоректні дані від Binance для {symbol}")
        return {
            "exchange": self.name,
            "symbol": symbol,
            "closes": [float(k[4]) for k in data],
            "volumes": [float(k[5]) for k in data],
            "current_price": float(data[-1][4])
        }


# --- ▼▼▼ ПОВНІСТЮ ЗАМІНІТЬ КЛАС BybitAdapter НА ЦЮ ВЕРСІЮ ▼▼▼ ---
class BybitAdapter(ExchangeAdapter):
    """Адаптер для біржі Bybit."""

    def __init__(self):
        super().__init__()
        self.name = "Bybit"
        # Для публічних даних (як klines) ключ не потрібен. Використовуємо асинхронний HTTP клієнт.
        self.client = HTTP(testnet=False)
        self.base_url = "https://api.bybit.com"

    async def get_klines(self, session, symbol, interval='60', limit=100):  # '60' хвилин = 1 година
        # Викликаємо метод для отримання klines
        response = await self.client.get_kline(
            category="spot",  # Вказуємо, що нас цікавить спотовий ринок
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        # Обробляємо відповідь
        return self.format_klines_data(response.get('result', {}).get('list', []), symbol)

    async def get_market_tickers(self, session):  # session тут більше не потрібен, але залишимо для сумісності
        response = await self.client.get_tickers(category="spot")
        return response.get('result', {}).get('list', [])

    def format_klines_data(self, data, symbol):
        if not isinstance(data, list) or not data:
            raise ValueError(f"Некоректні дані від Bybit для {symbol}")

        # В новому API дані приходять вже в правильному порядку (від старих до нових)
        # [startTime, open, high, low, close, volume, turnover]
        return {
            "exchange": self.name,
            "symbol": symbol,
            "closes": [float(k[4]) for k in data],
            "volumes": [float(k[5]) for k in data],
            "current_price": float(data[-1][4])
        }


# Створюємо словник з нашими адаптерами для легкого доступу
EXCHANGES = {
    "Binance": BinanceAdapter(),
    "Bybit": BybitAdapter()
}
# ----------------------------------------------------

user_coins = {}
PAGE_SIZE = 30


# (Функції calculate_rsi, calculate_ema, get_usdt_pairs, get_page, build_coin_keyboard, get_binance_data, get_account_balance залишаються БЕЗ ЗМІН)
# ... (скопіюйте їх зі свого попереднього коду)
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        upval = max(delta, 0)
        downval = -min(delta, 0)
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi[-1]


def calculate_ema(prices, period=10):
    ema = [np.mean(prices[:period])]
    k = 2 / (period + 1)
    for price in prices[period:]:
        ema.append(price * k + ema[-1] * (1 - k))
    return ema[-1]


# --- ▼▼▼ ПОВНІСТЮ ЗАМІНІТЬ ВАШУ ФУНКЦІЮ run_market_scanner_for_exchange НА ЦЮ ▼▼▼ ---
async def run_market_scanner_for_exchange(session, adapter):
    """
    Запускає сканер ринку для КОНКРЕТНОЇ біржі через її адаптер.
    """
    logger.info(f"Запускаю сканер ринку для біржі {adapter.name}...")
    promising_coins = set()

    try:
        all_tickers = await adapter.get_market_tickers(session)

        for ticker in all_tickers:
            symbol = None
            quote_volume = 0
            price_change_percent = 0

            # --- Адаптуємо логіку під унікальну структуру даних кожної біржі ---
            if adapter.name == "Binance":
                symbol = ticker.get('symbol')
                if not symbol or not symbol.endswith('USDT'): continue
                quote_volume = float(ticker.get('quoteVolume', 0))
                price_change_percent = float(ticker.get('priceChangePercent', 0))

            # --- ▼▼▼ ОСЬ ВИПРАВЛЕНА ЛОГІКА ДЛЯ BYBIT ▼▼▼ ---
            elif adapter.name == "Bybit":
                symbol = ticker.get('symbol')
                if not symbol or not symbol.endswith('USDT'): continue

                # Правильна назва поля для зміни ціни - 'price_24h_pcnt'
                # Воно повертає значення у вигляді десяткового дробу (напр., 0.05 для 5%), тому множимо на 100
                price_change_percent = float(ticker.get('price_24h_pcnt', 0)) * 100

                # Правильна назва поля для об'єму в USDT - 'turnover_24h'
                quote_volume = float(ticker.get('turnover_24h', 0))

            if not symbol: continue

            # Застосовуємо наші стандартні фільтри
            if quote_volume > 2000000 and abs(price_change_percent) > 5:
                promising_coins.add(f"{adapter.name}:{symbol}")

        logger.info(f"Сканер знайшов {len(promising_coins)} перспективних монет на {adapter.name}.")
        return promising_coins

    except Exception as e:
        logger.error(f"Помилка в роботі сканера для {adapter.name}: {e}")
        return set()

async def get_usdt_pairs(session):
    url = f"{BASE_URL}/api/v3/exchangeInfo"
    async with session.get(url) as resp:
        data = await resp.json()
    pairs = [
        s["symbol"] for s in data["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
    ]
    return sorted(pairs)


def get_page(coins, page: int):
    start = page * PAGE_SIZE
    end = start + PAGE_SIZE
    return coins[start:end]


# --- ▼▼▼ ОНОВЛЕНА ФУНКЦІЯ ДЛЯ СТВОРЕННЯ КЛАВІАТУР ▼▼▼ ---
def build_coin_keyboard(coins, page, action, all_count):
    # Створюємо кнопки для кожної монети на сторінці
    keyboard = [
        [InlineKeyboardButton(c, callback_data=f"{action}_{c}")]
        for c in coins
    ]

    # Створюємо рядок з кнопками навігації (пагінації)
    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton("⬅️ Назад", callback_data=f"page_{action}_{page - 1}"))
    if (page + 1) * PAGE_SIZE < all_count:
        nav_buttons.append(InlineKeyboardButton("➡️ Вперед", callback_data=f"page_{action}_{page + 1}"))

    # Якщо є кнопки навігації, додаємо їх як один рядок
    if nav_buttons:
        keyboard.append(nav_buttons)

    # --- НОВА ЧАСТИНА: Додаємо рядок з кнопкою "Головне меню" ---
    # Ця кнопка буде з'являтися завжди внизу
    keyboard.append([InlineKeyboardButton("🏠 Головне меню", callback_data="back_to_start")])

    return InlineKeyboardMarkup(keyboard)


async def get_account_balance(session):
    try:
        timestamp = int(time.time() * 1000)
        query_string = f"timestamp={timestamp}"
        signature = hmac.new(BINANCE_API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        url = f"{BASE_URL}/api/v3/account?{query_string}&signature={signature}"
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        async with session.get(url, headers=headers) as resp:
            data = await resp.json()
        balances = {item['asset']: float(item['free']) for item in data.get('balances', []) if float(item['free']) > 0}
        return balances
    except Exception as e:
        logger.error(f"Помилка отримання балансу: {e}")
        return {}


# --- ▼▼▼ ПОВНІСТЮ ЗАМІНІТЬ СТАРУ ФУНКЦІЮ НА ЦЮ АСИНХРОННУ ВЕРСІЮ ▼▼▼ ---
async def get_sentiment_analysis(session, asset_name):
    # ... (код цієї функції залишається без змін)
    logger.info(f"Запускаю асинхронний аналіз настроїв для {asset_name}...")
    combined_texts = []
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories={asset_name}"
        async with session.get(url) as resp:
            news_data = await resp.json()
            if news_data.get('Data'):
                headlines = [article['title'] for article in news_data['Data'][:10]]
                combined_texts.extend(headlines)
                logger.info(f"Отримано {len(headlines)} новин з CryptoCompare.")
    except Exception as e:
        logger.error(f"Помилка отримання новин з CryptoCompare: {e}")
    try:
        subreddit = await reddit.subreddit("CryptoCurrency")
        search_query = f"title:{asset_name}"
        posts = [post.title async for post in subreddit.search(search_query, sort="hot", limit=10)]
        if posts:
            combined_texts.extend(posts)
            logger.info(f"Отримано {len(posts)} постів з Reddit.")
    except Exception as e:
        logger.error(f"Помилка отримання постів з Reddit: {e}")
    if not combined_texts:
        logger.warning(f"Не вдалося зібрати текст для аналізу {asset_name}.")
        return 0.0, ""
    full_text = ". ".join(combined_texts)
    sentiment_score = sia.polarity_scores(full_text)['compound']
    logger.info(f"Фінальна оцінка настроїв для {asset_name}: {sentiment_score:.2f}")
    # Повертаємо і оцінку, і сам текст новин для LLM
    return sentiment_score, full_text


# --- ▼▼▼ ТРОХИ ОНОВІТЬ ФУНКЦІЮ analyze_coin ▼▼▼ ---
# Повністю замініть вашу стару функцію analyze_coin на цю
# --- ▼▼▼ ПОВНІСТЮ ЗАМІНІТЬ ВАШУ analyze_coin НА ЦЮ НОВУ ВЕРСІЮ ▼▼▼ ---
async def analyze_coin(session, symbol, exchange_name, balances):
    try:
        # Отримуємо адаптер за назвою біржі
        adapter = EXCHANGES.get(exchange_name)
        if not adapter:
            raise ValueError(f"Адаптер для біржі {exchange_name} не знайдено.")

        # Отримуємо дані через адаптер
        market_data = await adapter.get_klines(session, symbol)

        closes = market_data["closes"]
        volumes = market_data["volumes"]
        price = market_data["current_price"]

        rsi = calculate_rsi(np.array(closes))
        ema10 = calculate_ema(np.array(closes), 10)
        ema50 = calculate_ema(np.array(closes), 50)
        vol_trend = "зростає" if volumes[-1] > np.mean(volumes[-10:]) else "падає"
        asset = symbol.replace("USDT", "")
        balance = balances.get(asset, 0)

        # --- ЕТАП 1: ШВИДКИЙ ПОПЕРЕДНІЙ АНАЛІЗ (VADER) ---
        vader_score, news_text = await get_sentiment_analysis(session, asset)

        preliminary_signal = False
        if rsi < 35 and ema10 > ema50 and vol_trend == "зростає" and vader_score >= 0.1:
            preliminary_signal = True
        elif rsi > 65 and ema10 < ema50 and vader_score <= -0.1:
            preliminary_signal = True

        if not preliminary_signal:
            return {
                "exchange": exchange_name, "symbol": symbol, "price": price, "rsi": rsi,
                "recommendation": "⚪️ NEUTRAL (Немає сильних сигналів)",
                "balance": balance, "stop_loss": None, "take_profit": None
            }

        # --- ЕТАП 2: ГЛИБОКИЙ АНАЛІЗ (LLM) ---
        logger.info(f"Попередній сигнал знайдено для {symbol} на {exchange_name}. Запускаю LLM-аналіз...")

        if model is None:
            return {"recommendation": "NEUTRAL", "reason": "AI model is not available."}

        # ... (решта логіки з LLM залишається без змін)
        prompt = f"""
        You are an expert crypto market analyst. Analyze the following data for {symbol} on {exchange_name} and provide a trading recommendation.
        Technical Indicators: - RSI: {rsi:.2f} - EMA Trend: {'Bullish' if ema10 > ema50 else 'Bearish'} - Volume Trend: {vol_trend}
        Recent News and Discussions: {news_text}
        Based on all information, provide your analysis as a single JSON object:
        {{"recommendation": "BUY" or "SELL" or "NEUTRAL", "confidence": "LOW" or "MEDIUM" or "HIGH", "reason": "Brief explanation."}}
        """
        try:
            response = await model.generate_content_async(prompt)
            cleaned_response = response.text.replace("```json", "").replace("```", "").strip()
            llm_result = json.loads(cleaned_response)
        except Exception as e:
            logger.error(f"Помилка LLM-аналізу для {symbol}: {e}")
            llm_result = {"recommendation": "NEUTRAL", "reason": "Error during AI analysis."}

        recommendation = f"⚪️ NEUTRAL ({llm_result.get('reason', 'N/A')})"
        stop_loss, take_profit = None, None
        if llm_result.get('recommendation') == "BUY" and llm_result.get('confidence') in ["MEDIUM", "HIGH"]:
            recommendation = f"🟢 BUY (Підтверджено ШІ. Впевненість: {llm_result.get('confidence')})"
            stop_loss = price * 0.98
            take_profit = price * 1.05
        elif llm_result.get('recommendation') == "SELL" and llm_result.get('confidence') in ["MEDIUM", "HIGH"]:
            recommendation = f"🔴 SELL (Підтверджено ШІ. Впевненість: {llm_result.get('confidence')})"

        return {
            "exchange": exchange_name, "symbol": symbol, "price": price, "rsi": rsi,
            "recommendation": recommendation, "balance": balance,
            "stop_loss": stop_loss, "take_profit": take_profit
        }

    except Exception as e:
        logger.error(f"Помилка аналізу {symbol} на {exchange_name}: {e}")
        return None

# --- Функції start та monitor залишаються майже без змін, але ми оновимо текст повідомлень в них ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_coins.setdefault(user_id, [])

    # --- ▼▼▼ ЗМІНЕНО: ДОДАНО КНОПКУ СКАНЕРА ▼▼▼ ---
    keyboard = [
        [InlineKeyboardButton("🔍 Сканер ринку 🔍", callback_data="market_scanner")],
        [InlineKeyboardButton("➕ Додати монету ➕", callback_data="add")],
        [InlineKeyboardButton("➖ Видалити монету ➖", callback_data="remove")],
        [InlineKeyboardButton("📋 Мої монети 📋", callback_data="mycoins")],
    ]
    await update.message.reply_text(
        f"Привіт 👋! Я твій крипто-помічник.\n\n"
        f"• Натисни **'Сканер ринку'**, щоб знайти найактивніші монети прямо зараз.\n"
        f"• Або керуй своїм персональним списком відстеження.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )


# --- ▼▼▼ ПОВНІСТЮ ЗАМІНІТЬ ВАШУ ФУНКЦІЮ button_handler НА ЦЮ ▼▼▼ ---
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id

    async with aiohttp.ClientSession() as session:
        # Обробка додавання монет (зі сканера або загального списку)
        if query.data.startswith("scanner_add_") or query.data.startswith("addcoin_"):
            if query.data.startswith("scanner_add_"):
                coin_identifier = query.data.replace("scanner_add_", "")
            else:  # addcoin_
                # Для загального списку ми за замовчуванням вважаємо, що це Binance
                coin_symbol = query.data.replace("addcoin_", "")
                coin_identifier = f"Binance:{coin_symbol}"

            # Зараз ми зберігаємо тільки символ, без біржі. Це потрібно буде вдосконалити в майбутньому.
            symbol_to_add = coin_identifier.split(':')[1]
            user_coins.setdefault(user_id, [])

            if symbol_to_add not in user_coins[user_id]:
                user_coins[user_id].append(symbol_to_add)
                await query.answer(text=f"✅ {symbol_to_add} додано до списку!", show_alert=False)
            else:
                await query.answer(text=f"⚠️ {symbol_to_add} вже є у вашому списку.", show_alert=False)
            return

        # Для всіх інших випадків відповідаємо на початку
        await query.answer()

        # --- БЛОК СКАНЕРА РИНКУ ---
        if query.data == "market_scanner":
            await query.edit_message_text("⏳ Сканую ринки Binance та Bybit...")

            all_promising_coins = set()
            for exchange_name, adapter in EXCHANGES.items():
                promising_on_exchange = await run_market_scanner_for_exchange(session, adapter)
                all_promising_coins.update(promising_on_exchange)

            if not all_promising_coins:
                keyboard = [[InlineKeyboardButton("🏠 Головне меню", callback_data="back_to_start")]]
                await query.edit_message_text("Наразі на ринках немає активних монет.",
                                              reply_markup=InlineKeyboardMarkup(keyboard))
                return

            keyboard = []
            for coin_identifier in sorted(list(all_promising_coins)):
                exchange_name, symbol = coin_identifier.split(':')
                button = [InlineKeyboardButton(f"➕ {exchange_name}: {symbol}",
                                               callback_data=f"scanner_add_{coin_identifier}")]
                keyboard.append(button)

            keyboard.append([InlineKeyboardButton("🏠 Головне меню", callback_data="back_to_start")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            message = "📈 **Результати сканера ринків:**\n\nНатисніть на монету, щоб додати її до списку відстеження:"
            await query.edit_message_text(text=message, reply_markup=reply_markup)

        # --- БЛОК АНАЛІЗУ КОНКРЕТНОЇ МОНЕТИ ---
        elif query.data.startswith("analyze_"):
            coin = query.data.replace("analyze_", "")
            # За замовчуванням, аналіз вручну йде по Binance. Це можна розширити в майбутньому.
            exchange_to_analyze = "Binance"
            await query.edit_message_text(f"⏳ Роблю глибокий аналіз {coin} на {exchange_to_analyze}...")

            balances = await get_account_balance(session)
            analysis_data = await analyze_coin(session, coin, exchange_to_analyze, balances)

            keyboard = [[InlineKeyboardButton("⬅️ До списку", callback_data="mycoins")],
                        [InlineKeyboardButton("🏠 Головне меню", callback_data="back_to_start")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            if not analysis_data:
                await query.edit_message_text(f"Не вдалося отримати дані для {coin}.", reply_markup=reply_markup)
                return

            message = (
                f"📊 **{analysis_data.get('exchange')} | {analysis_data.get('symbol')}**\n\n"
                f"💰 Поточна ціна: `{analysis_data.get('price', 0):.6f}`\n"
                f"📈 RSI: `{analysis_data.get('rsi', 0):.2f}`\n\n"
                f"📌 **Сигнал від ШІ:** {analysis_data.get('recommendation', 'Помилка')}"
            )
            if analysis_data.get("stop_loss"):
                message += f"\n\n**Пропонований план:**\n🛡️ Stop-Loss: `{analysis_data['stop_loss']:.6f}`\n🎯 Take-Profit: `{analysis_data['take_profit']:.6f}`"

            await query.edit_message_text(text=message, reply_markup=reply_markup, parse_mode='Markdown')

        # ... (решта блоків: back_to_start, add, remove, mycoins і т.д. залишаються без змін)
        elif query.data == "back_to_start":
            keyboard = [[InlineKeyboardButton("🔍 Сканер ринку", callback_data="market_scanner")],
                        [InlineKeyboardButton("➕ Додати монету", callback_data="add")],
                        [InlineKeyboardButton("➖ Видалити монету", callback_data="remove")],
                        [InlineKeyboardButton("📋 Мої монети", callback_data="mycoins")], ]
            await query.edit_message_text(f"Привіт 👋! Я твій крипто-помічник.",
                                          reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        elif query.data == "add":
            all_pairs = await get_usdt_pairs(session)
            page = 0
            coins_page = get_page(all_pairs, page)
            reply_markup = build_coin_keyboard(coins_page, page, "addcoin", len(all_pairs))
            await query.edit_message_text("Оберіть монету для додавання:", reply_markup=reply_markup)
        elif query.data.startswith("page_addcoin_"):
            page = int(query.data.replace("page_addcoin_", ""))
            all_pairs = await get_usdt_pairs(session)
            coins_page = get_page(all_pairs, page)
            reply_markup = build_coin_keyboard(coins_page, page, "addcoin", len(all_pairs))
            await query.edit_message_text("Оберіть монету для додавання:", reply_markup=reply_markup)
        elif query.data == "remove":
            coins = user_coins.get(user_id, [])
            if not coins: await query.edit_message_text("Список порожній."); return
            page = 0
            coins_page = get_page(coins, page)
            reply_markup = build_coin_keyboard(coins_page, page, "removecoin", len(coins))
            await query.edit_message_text("Оберіть монету для видалення:", reply_markup=reply_markup)
        elif query.data.startswith("page_removecoin_"):
            page = int(query.data.replace("page_removecoin_", ""))
            coins = user_coins.get(user_id, [])
            coins_page = get_page(coins, page)
            reply_markup = build_coin_keyboard(coins_page, page, "removecoin", len(coins))
            await query.edit_message_text("Оберіть монету для видалення:", reply_markup=reply_markup)
        elif query.data.startswith("removecoin_"):
            coin = query.data.replace("removecoin_", "")
            if coin in user_coins.get(user_id, []):
                user_coins[user_id].remove(coin); await query.edit_message_text(f"❌ {coin} видалено")
            else:
                await query.edit_message_text(f"⚠️ {coin} немає у списку")
            await asyncio.sleep(2)
            coins = user_coins.get(user_id, [])
            if not coins: keyboard = [
                [InlineKeyboardButton("🏠 Головне меню", callback_data="back_to_start")]]; await query.edit_message_text(
                "Список порожній.", reply_markup=InlineKeyboardMarkup(keyboard)); return
            page = 0;
            coins_page = get_page(coins, page);
            reply_markup = build_coin_keyboard(coins_page, page, "removecoin", len(coins));
            await query.edit_message_text("Оберіть монету для видалення:", reply_markup=reply_markup)
        elif query.data == "mycoins":
            coins = user_coins.get(user_id, [])
            if not coins: await query.edit_message_text("Список відстежуваних монет порожній."); return
            keyboard = [[InlineKeyboardButton(c, callback_data=f"analyze_{c}")] for c in coins]
            keyboard.append([InlineKeyboardButton("🏠 Головне меню", callback_data="back_to_start")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text("📋 Твої монети (натисніть для глибокого аналізу на Binance):",
                                          reply_markup=reply_markup)


# --- ▼▼▼ ПОВНІСТЮ ЗАМІНІТЬ ВАШУ ФУНКЦІЮ monitor НА ЦЮ ОНОВЛЕНУ ВЕРСІЮ ▼▼▼ ---
async def monitor(app):
    async with aiohttp.ClientSession() as session:
        while True:
            # --- КРОК 1: ЗАПУСКАЄМО СКАНЕР ДЛЯ КОЖНОЇ БІРЖІ ---
            all_promising_coins = set()
            scanner_results_summary = []  # Створюємо список для зведення результатів сканера

            for exchange_name, adapter in EXCHANGES.items():
                promising_on_exchange = await run_market_scanner_for_exchange(session, adapter)
                all_promising_coins.update(promising_on_exchange)
                # Додаємо рядок у наше зведення
                scanner_results_summary.append(
                    f"• {exchange_name}: знайдено {len(promising_on_exchange)} активних монет.")

            # --- КРОК 2: АНАЛІЗУЄМО МОНЕТИ ТА НАДСИЛАЄМО СИГНАЛИ ---
            # Ця частина залишається майже без змін
            for user_id, user_tracked_coins in user_coins.items():
                balances = await get_account_balance(session)
                coins_to_analyze = {f"Binance:{coin}" for coin in user_tracked_coins} | all_promising_coins

                if not coins_to_analyze: continue

                logger.info(f"Для користувача {user_id} аналізується {len(coins_to_analyze)} монет.")
                signal_messages = []  # Перейменовуємо, щоб не плутати з повідомленням про сканування

                for coin_identifier in coins_to_analyze:
                    exchange_name, symbol = coin_identifier.split(':')
                    analysis_data = await analyze_coin(session, symbol, exchange_name, balances)
                    await asyncio.sleep(2)

                    if analysis_data and (
                            "BUY" in analysis_data["recommendation"] or "SELL" in analysis_data["recommendation"]):
                        is_personal = f"Binance:{symbol}" in {f"Binance:{c}" for c in user_tracked_coins}
                        alert_type = "🚨 **Сигнал по вашій монеті!** 🚨" if is_personal else "🔥 **Сигнал зі сканера ринку!** 🔥"
                        message = (
                            f"{alert_type}\n\n"
                            f"**Біржа: `{analysis_data['exchange']}`**\n"
                            f"Монета: **{analysis_data['symbol']}**\n"
                            f"💰 Ціна: `{analysis_data['price']:.6f}` USDT\n"
                            f"📌 Сигнал від ШІ: **{analysis_data['recommendation']}**"
                        )
                        if analysis_data.get("stop_loss"):
                            message += f"\n🛡️ Stop-Loss: `{analysis_data['stop_loss']:.6f}`\n🎯 Take-Profit: `{analysis_data['take_profit']:.6f}`"
                        signal_messages.append(message)

                # Надсилаємо сильні сигнали, ЯКЩО вони є
                if signal_messages:
                    try:
                        full_message = "\n\n".join(signal_messages)
                        # Це сповіщення буде гучним
                        await app.bot.send_message(chat_id=user_id, text=full_message, parse_mode='Markdown')
                    except Exception as e:
                        logger.error(f"Помилка відправки СИГНАЛУ {user_id}: {e}")

            # --- КРОК 3: НАДСИЛАЄМО ІНФОРМАЦІЙНЕ ЗВЕДЕННЯ ВСІМ КОРИСТУВАЧАМ ---
            if user_coins:  # Надсилаємо, тільки якщо є хоча б один активний користувач
                summary_text = (
                        f"**📈 Результати планового сканування:**\n\n"
                        + "\n".join(scanner_results_summary)
                        + "\n\n*Бот продовжує моніторинг. Сильні сигнали `BUY` або `SELL` будуть надіслані окремим повідомленням.*"
                )

                # Проходимо по всіх користувачах і надсилаємо тихе сповіщення
                for user_id in user_coins.keys():
                    try:
                        # disable_notification=True робить сповіщення тихим
                        await app.bot.send_message(chat_id=user_id, text=summary_text, parse_mode='Markdown',
                                                   disable_notification=True)
                    except Exception as e:
                        logger.error(f"Помилка відправки ЗВЕДЕННЯ {user_id}: {e}")

            logger.info("Цикл моніторингу завершено. Наступна перевірка за 15 хвилин.")
            await asyncio.sleep(900)


# Основна функція (без змін)
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))

    async def on_startup(app):
        asyncio.create_task(monitor(app))

    app.post_init = on_startup
    app.run_polling()


if __name__ == "__main__":
    main()