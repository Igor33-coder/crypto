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
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# --- ▼▼▼ НОВІ ІМПОРТИ ДЛЯ ШІ ТА НОВИН ▼▼▼ ---
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- ▼▼▼ ПЕРВИННЕ НАЛАШТУВАННЯ NLTK ▼▼▼ ---
# Це завантаження потрібно виконати лише один раз.
# Якщо виникає помилка, розкоментуйте наступний рядок, запустіть скрипт,
# а після успішного завантаження знову закоментуйте.
# nltk.download('vader_lexicon')
# ----------------------------------------------------

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
# Налаштовуємо модель
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash') # Використовуємо швидку і дешеву модель
# ----------------------------------------------------

# --------------------------
# Ініціалізація клієнтів
sia = SentimentIntensityAnalyzer()
# --------------------------

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


# ... (весь ваш попередній код залишається без змін)

# --- ▼▼▼ НОВА ФУНКЦІЯ-СКРИНЕР ▼▼▼ ---
async def market_screener(session):
    """
    Робить один запит до Binance, щоб отримати статистику за 24 години
    і відфільтрувати найцікавіші монети для подальшого аналізу.
    """
    logger.info("Запускаю скринер ринку...")
    promising_coins = set()  # Використовуємо set для швидкого пошуку
    url = f"{BASE_URL}/api/v3/ticker/24hr"

    try:
        async with session.get(url) as resp:
            all_tickers = await resp.json()

        for ticker in all_tickers:
            symbol = ticker['symbol']
            # --- Критерії фільтрації ---
            # 1. Тільки пари до USDT
            # 2. Об'єм торгів > 2,000,000 USDT за 24 години
            # 3. Ціна змінилася більше ніж на 5% в будь-яку сторону
            if (symbol.endswith('USDT') and
                    float(ticker['quoteVolume']) > 2000000 and
                    abs(float(ticker['priceChangePercent'])) > 5):
                promising_coins.add(symbol)

        logger.info(f"Скринер знайшов {len(promising_coins)} перспективних монет.")
        return promising_coins

    except Exception as e:
        logger.error(f"Помилка в роботі скринера ринку: {e}")
        return set()  # Повертаємо пустий набір у разі помилки

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


async def get_binance_data(session, symbol="DOGEUSDT", interval="1h", limit=100):
    url = f"{BASE_URL}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    async with session.get(url) as resp:
        data = await resp.json()
    closes = [float(k[4]) for k in data]
    volumes = [float(k[5]) for k in data]
    price = float(data[-1][4])
    return closes, volumes, price


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


# --------------------------
# --- ▼▼▼ НОВА ФУНКЦІЯ ДЛЯ АНАЛІЗУ НОВИН ▼▼▼ ---
# --------------------------
# --- ▼▼▼ ПОВНІСТЮ ЗАМІНІТЬ СТАРУ ФУНКЦІЮ НА ЦЮ НОВУ ▼▼▼ ---
# --- ▼▼▼ ПОВНІСТЮ ЗАМІНІТЬ СТАРУ ФУНКЦІЮ НА ЦЮ АСИНХРОННУ ВЕРСІЮ ▼▼▼ ---
# --- ▼▼▼ ЗАМІСТЬ get_sentiment_analysis ВСТАВТЕ ЦЮ НОВУ ФУНКЦІЮ ▼▼▼ ---
async def get_llm_analysis(session, symbol, rsi, ema_bullish, volume_trend):
    """
    Збирає новини, формує комплексний запит до LLM (Gemini)
    і отримує глибокий аналіз ринку.
    """
    asset_name = symbol.replace("USDT", "")
    logger.info(f"Запускаю LLM-аналіз для {asset_name}...")

    # 1. Збираємо текстові дані (новини та пости)
    # Цей код взято з вашої попередньої get_sentiment_analysis
    combined_texts = []
    try:
        # CryptoCompare
        url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories={asset_name}"
        async with session.get(url) as resp:
            news_data = await resp.json()
            if news_data.get('Data'):
                headlines = [article['title'] for article in news_data['Data'][:5]]  # Беремо 5 новин
                combined_texts.extend(headlines)
        # Reddit
        subreddit = await reddit.subreddit("CryptoCurrency")
        posts = [post.title async for post in
                 subreddit.search(f"title:{asset_name}", sort="hot", limit=5)]  # Беремо 5 постів
        combined_texts.extend(posts)
    except Exception as e:
        logger.error(f"Помилка збору даних для LLM: {e}")

    if not combined_texts:
        return {"recommendation": "NEUTRAL", "reason": "Could not fetch market data."}

    news_string = "\n".join(f"- {text}" for text in combined_texts)

    # 2. Формуємо потужний промпт для Gemini
    prompt = f"""
    You are an expert crypto market analyst. Analyze the following data for {symbol} and provide a trading recommendation.

    Technical Indicators:
    - RSI: {rsi:.2f}
    - EMA Trend: {'Bullish' if ema_bullish else 'Bearish'}
    - Volume Trend: {volume_trend}

    Recent News and Discussions:
    {news_string}

    Based on the synthesis of ALL the above information, provide your analysis.
    Your response MUST be a single, valid JSON object with the following structure:
    {{
      "recommendation": "BUY", "SELL", or "NEUTRAL",
      "confidence": "LOW", "MEDIUM", or "HIGH",
      "reason": "A brief, one-sentence explanation for your recommendation."
    }}
    """

    # 3. Робимо запит до API і обробляємо відповідь
    try:
        response = await model.generate_content_async(prompt)
        # Очищуємо відповідь від markdown форматування
        cleaned_response = response.text.replace("```json", "").replace("```", "").strip()
        analysis_json = json.loads(cleaned_response)
        logger.info(f"LLM-аналіз для {asset_name} успішний: {analysis_json}")
        return analysis_json
    except Exception as e:
        logger.error(f"Помилка LLM-аналізу для {asset_name}: {e}")
        return {"recommendation": "NEUTRAL", "reason": "Error during AI analysis."}


# --- ▼▼▼ ТРОХИ ОНОВІТЬ ФУНКЦІЮ analyze_coin ▼▼▼ ---
async def analyze_coin(session, symbol, balances):
    try:
        closes, volumes, price = await get_binance_data(session, symbol)
        rsi = calculate_rsi(np.array(closes))
        ema10 = calculate_ema(np.array(closes), 10)
        ema50 = calculate_ema(np.array(closes), 50)
        vol_trend = "зростає" if volumes[-1] > np.mean(volumes[-10:]) else "падає"
        asset = symbol.replace("USDT", "")
        balance = balances.get(asset, 0)

        # --- ІНТЕГРАЦІЯ LLM ---
        # Передаємо ключові технічні дані в нову функцію
        llm_result = await get_llm_analysis(session, symbol, rsi, ema10 > ema50, vol_trend)

        # Формуємо фінальний результат на основі відповіді LLM
        recommendation = f"⚪️ NEUTRAL ({llm_result.get('reason', 'N/A')})"
        stop_loss, take_profit = None, None

        if llm_result.get('recommendation') == "BUY" and llm_result.get('confidence') in ["MEDIUM", "HIGH"]:
            recommendation = f"🟢 BUY (Впевненість: {llm_result.get('confidence')}. Причина: {llm_result.get('reason')})"
            stop_loss = price * 0.98
            take_profit = price * 1.05
        elif llm_result.get('recommendation') == "SELL" and llm_result.get('confidence') in ["MEDIUM", "HIGH"]:
            recommendation = f"🔴 SELL (Впевненість: {llm_result.get('confidence')}. Причина: {llm_result.get('reason')})"

        return {
            "symbol": symbol, "price": price, "rsi": rsi,
            "recommendation": recommendation,
            "balance": balance,
            "stop_loss": stop_loss, "take_profit": take_profit
        }

    except Exception as e:
        logger.error(f"Помилка аналізу {symbol}: {e}")
        return None

# --- Функції start та monitor залишаються майже без змін, але ми оновимо текст повідомлень в них ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_coins.setdefault(user_id, [])

    # --- ▼▼▼ ЗМІНЕНО: ДОДАНО КНОПКУ СКАНЕРА ▼▼▼ ---
    keyboard = [
        [InlineKeyboardButton("🔍 Сканер ринку", callback_data="market_scanner")],
        [InlineKeyboardButton("➕ Додати монету", callback_data="add")],
        [InlineKeyboardButton("➖ Видалити монету", callback_data="remove")],
        [InlineKeyboardButton("📋 Мої монети", callback_data="mycoins")],
    ]
    await update.message.reply_text(
        f"Привіт 👋! Я твій крипто-помічник.\n\n"
        f"• Натисни **'Сканер ринку'**, щоб знайти найактивніші монети прямо зараз.\n"
        f"• Або керуй своїм персональним списком відстеження.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

# --------------------------
# --- ▼▼▼ ОНОВЛЕНА ФУНКЦІЯ ОБРОБКИ КНОПОК ▼▼▼ ---
# --- (Змінено лише блок analyze_, щоб показувати дані ШІ) ---
# --------------------------
# Повністю замініть вашу стару функцію button_handler на цю
# Повністю замініть вашу стару функцію button_handler на цю
# Повністю замініть вашу стару функцію button_handler на цю
# Повністю замініть вашу стару функцію button_handler на цю
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id

    async with aiohttp.ClientSession() as session:
        # --- БЛОК СКАНЕРА РИНКУ ---
        if query.data == "market_scanner":
            await query.answer()
            await query.edit_message_text("⏳ Шукаю активні монети на ринку...")

            promising_coins = await market_screener(session)

            if not promising_coins:
                keyboard = [[InlineKeyboardButton("🏠 Головне меню", callback_data="back_to_start")]]
                await query.edit_message_text(
                    "Наразі на ринку немає монет з високою волатильністю та об'ємом.",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
                return

            keyboard = []
            for coin in sorted(list(promising_coins)):
                button = [InlineKeyboardButton(f"➕ {coin}", callback_data=f"scanner_add_{coin}")]
                keyboard.append(button)

            keyboard.append([InlineKeyboardButton("🏠 Головне меню", callback_data="back_to_start")])
            reply_markup = InlineKeyboardMarkup(keyboard)

            message = (
                f"📈 **Результати сканера ринку:**\n\n"
                f"Ось список найактивніших монет. Натисніть на монету, щоб додати її до вашого списку відстеження:"
            )
            await query.edit_message_text(text=message, reply_markup=reply_markup)
            return

        # --- БЛОК ОБРОБКИ ДОДАВАННЯ ЗІ СКАНЕРА ---
        elif query.data.startswith("scanner_add_"):
            coin = query.data.replace("scanner_add_", "")
            user_coins.setdefault(user_id, [])

            if coin not in user_coins[user_id]:
                user_coins[user_id].append(coin)
                await query.answer(text=f"✅ {coin} додано до списку!", show_alert=False)
            else:
                await query.answer(text=f"⚠️ {coin} вже є у вашому списку.", show_alert=False)
            return

        # --- БЛОК ПОВЕРНЕННЯ В ГОЛОВНЕ МЕНЮ ---
        elif query.data == "back_to_start":
            await query.answer()
            keyboard = [
                [InlineKeyboardButton("🔍 Сканер ринку", callback_data="market_scanner")],
                [InlineKeyboardButton("➕ Додати монету", callback_data="add")],
                [InlineKeyboardButton("➖ Видалити монету", callback_data="remove")],
                [InlineKeyboardButton("📋 Мої монети", callback_data="mycoins")],
            ]
            await query.edit_message_text(
                f"Привіт 👋! Я твій крипто-помічник.\n\n"
                f"• Натисни **'Сканер ринку'**, щоб знайти найактивніші монети прямо зараз.\n"
                f"• Або керуй своїм персональним списком відстеження.",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )

        # --- БЛОК "ДОДАТИ МОНЕТУ" (ВХІД У ЗАГАЛЬНИЙ СПИСОК) ---
        elif query.data == "add":
            await query.answer()
            all_pairs = await get_usdt_pairs(session)
            page = 0
            coins_page = get_page(all_pairs, page)
            reply_markup = build_coin_keyboard(coins_page, page, "addcoin", len(all_pairs))
            await query.edit_message_text("Оберіть монету для додавання:", reply_markup=reply_markup)

        # --- БЛОК ПЕРЕГОРТАННЯ СТОРІНОК У ЗАГАЛЬНОМУ СПИСКУ ---
        elif query.data.startswith("page_addcoin_"):
            await query.answer()
            page = int(query.data.replace("page_addcoin_", ""))
            all_pairs = await get_usdt_pairs(session)
            coins_page = get_page(all_pairs, page)
            reply_markup = build_coin_keyboard(coins_page, page, "addcoin", len(all_pairs))
            await query.edit_message_text("Оберіть монету для додавання:", reply_markup=reply_markup)

        # --- ▼▼▼ ВИПРАВЛЕНИЙ БЛОК: ДОДАВАННЯ МОНЕТИ ІЗ ЗАГАЛЬНОГО СПИСКУ ▼▼▼ ---
        elif query.data.startswith("addcoin_"):
            coin = query.data.replace("addcoin_", "")
            user_coins.setdefault(user_id, [])

            if coin not in user_coins[user_id]:
                user_coins[user_id].append(coin)
                # Показуємо спливаюче сповіщення і залишаємось на місці
                await query.answer(text=f"✅ {coin} додано!", show_alert=False)
            else:
                # Показуємо спливаюче сповіщення і залишаємось на місці
                await query.answer(text=f"⚠️ {coin} вже є у списку.", show_alert=False)

            # Важливо! Ми більше не змінюємо екран, а просто завершуємо виконання.
            return

        # ... (решта коду залишається без змін)
        elif query.data == "remove":
            await query.answer()
            coins = user_coins.get(user_id, [])
            if not coins:
                await query.edit_message_text("Список порожній.")
                return
            page = 0
            coins_page = get_page(coins, page)
            reply_markup = build_coin_keyboard(coins_page, page, "removecoin", len(coins))
            await query.edit_message_text("Оберіть монету для видалення:", reply_markup=reply_markup)

        elif query.data.startswith("page_removecoin_"):
            await query.answer()
            page = int(query.data.replace("page_removecoin_", ""))
            coins = user_coins.get(user_id, [])
            coins_page = get_page(coins, page)
            reply_markup = build_coin_keyboard(coins_page, page, "removecoin", len(coins))
            await query.edit_message_text("Оберіть монету для видалення:", reply_markup=reply_markup)

        elif query.data.startswith("removecoin_"):
            await query.answer()
            coin = query.data.replace("removecoin_", "")
            if coin in user_coins.get(user_id, []):
                user_coins[user_id].remove(coin)
                await query.edit_message_text(f"❌ {coin} видалено")
            else:
                await query.edit_message_text(f"⚠️ {coin} немає у списку")
            await asyncio.sleep(2)
            coins = user_coins.get(user_id, [])
            if not coins:
                keyboard = [[InlineKeyboardButton("🏠 Головне меню", callback_data="back_to_start")]]
                await query.edit_message_text("Список порожній.", reply_markup=InlineKeyboardMarkup(keyboard))
                return
            page = 0
            coins_page = get_page(coins, page)
            reply_markup = build_coin_keyboard(coins_page, page, "removecoin", len(coins))
            await query.edit_message_text("Оберіть монету для видалення:", reply_markup=reply_markup)

        elif query.data == "mycoins":
            await query.answer()
            coins = user_coins.get(user_id, [])
            if not coins:
                await query.edit_message_text("Список відстежуваних монет порожній.")
                return
            keyboard = [[InlineKeyboardButton(c, callback_data=f"analyze_{c}")] for c in coins]
            keyboard.append([InlineKeyboardButton("🏠 Головне меню", callback_data="back_to_start")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text("📋 Твої монети (натисніть для глибокого аналізу):", reply_markup=reply_markup)

        elif query.data.startswith("analyze_"):
            await query.answer()
            coin = query.data.replace("analyze_", "")
            await query.edit_message_text(f"⏳ Роблю глибокий аналіз {coin}...")
            balances = await get_account_balance(session)
            analysis_data = await analyze_coin(session, coin, balances)
            keyboard = [
                [
                    InlineKeyboardButton("⬅️ До списку", callback_data="mycoins"),
                    InlineKeyboardButton("🏠 Головне меню", callback_data="back_to_start")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            if not analysis_data:
                await query.edit_message_text(f"Не вдалося отримати дані для {coin}.", reply_markup=reply_markup)
                return

            asset_name = analysis_data["symbol"].replace("USDT", "")
            message = (
                f"📊 **{analysis_data['symbol']}**\n\n"
                f"💰 Ціна: `{analysis_data['price']:.6f}`\n"
                f"📰 Настрій: **{analysis_data['sentiment_label']}** (`{analysis_data['sentiment_score']:.2f}`)\n"
                f"📈 RSI: `{analysis_data['rsi']:.2f}`\n\n"
                f"📌 **Сигнал: {analysis_data['recommendation']}**"
            )
            if analysis_data["recommendation"].startswith("🟢 BUY"):
                message += f"\n\n**План:**\n🛡️ Stop-Loss: `{analysis_data['stop_loss']:.6f}`\n🎯 Take-Profit: `{analysis_data['take_profit']:.6f}`"
            await query.edit_message_text(text=message, reply_markup=reply_markup, parse_mode='Markdown')

# --- Оновлена функція моніторингу ---
# --- ▼▼▼ ОНОВЛЕНА ФУНКЦІЯ МОНІТОРИНГУ (ГІБРИДНА ВЕРСІЯ) ▼▼▼ ---
async def monitor(app):
    async with aiohttp.ClientSession() as session:
        while True:
            # 1. Запускаємо сканер ринку ОДИН РАЗ на початку циклу.
            # Це ефективно, бо не потрібно робити це для кожного користувача.
            promising_coins_from_screener = await market_screener(session)

            if not promising_coins_from_screener:
                logger.info("Скринер не знайшов активних монет.")

            # 2. Тепер проходимо по кожному користувачеві.
            for user_id, user_tracked_coins in user_coins.items():
                balances = await get_account_balance(session)

                # 3. ▼▼▼ КЛЮЧОВА ЗМІНА: СТВОРЮЄМО ОБ'ЄДНАНИЙ СПИСОК ▼▼▼
                # Використовуємо `set` для автоматичного видалення дублікатів.
                # Якщо монета є і в списку користувача, і в сканері, вона буде перевірена лише раз.
                coins_to_analyze = set(user_tracked_coins) | promising_coins_from_screener

                if not coins_to_analyze:
                    continue  # Якщо у користувача немає монет і сканер порожній, переходимо до наступного.

                logger.info(f"Для користувача {user_id} аналізується {len(coins_to_analyze)} монет.")
                messages = []
                # 4. Аналізуємо монети з ОБ'ЄДНАНОГО списку.
                for coin in coins_to_analyze:
                    analysis_data = await analyze_coin(session, coin, balances)

                    if analysis_data and "NEUTRAL" not in analysis_data["recommendation"]:
                        # Визначаємо, звідки прийшов сигнал, для більш інформативного повідомлення.
                        if coin in user_tracked_coins:
                            alert_type = "🚨 **Сигнал по вашій монеті!** 🚨"
                        else:
                            alert_type = "🔥 **Сигнал зі сканера ринку!** 🔥"

                        message = (
                            f"{alert_type}\n\n"
                            f"Монета: **{analysis_data['symbol']}**\n"
                            f"💰 Ціна: `{analysis_data['price']:.6f}` USDT\n"
                            f"📰 Настрій новин: **{analysis_data['sentiment_label']}**\n"
                            f"📌 Сигнал: **{analysis_data['recommendation']}**"
                        )
                        if analysis_data.get("stop_loss"):
                            message += (
                                f"\n🛡️ Stop-Loss: `{analysis_data['stop_loss']:.6f}`\n"
                                f"🎯 Take-Profit: `{analysis_data['take_profit']:.6f}`"
                            )
                        messages.append(message)

                # 5. Надсилаємо знайдені сигнали користувачеві.
                if messages:
                    try:
                        full_message = "\n\n".join(messages)
                        await app.bot.send_message(chat_id=user_id, text=full_message, parse_mode='Markdown')
                    except Exception as e:
                        logger.error(f"Помилка відправки {user_id}: {e}")

            logger.info("Цикл моніторингу завершено. Наступна перевірка за годину.")
            await asyncio.sleep(3600)


# --------------------------
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