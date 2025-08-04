import os
import telegram

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

async def send_telegram_alert(message: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram credentials not set. Skipping alert.")
        return
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text=message)
        print(f"Telegram alert sent: {message}")
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")