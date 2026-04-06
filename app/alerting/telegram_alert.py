import requests

from app.config import settings


def send_telegram_alert(message: str) -> bool:
    """Send an alert message to Telegram. Returns True on success."""
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return False

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    payload = {"chat_id": settings.telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    resp = requests.post(url, json=payload, timeout=10)
    return resp.ok
