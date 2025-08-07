import os
import telegram
import asyncio

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

async def send_telegram_message(message: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram credentials not set. Skipping message.")
        return
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
        print(f"Telegram message sent.")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")