import os
import logging
import asyncio
import base64
import io
import json  # Added for webhook parsing
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI
from PIL import Image
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# =========================
# Logging Configuration
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# FastAPI App
# =========================
app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Dark Bot (Multimodal Edition) is running! 🚀"}

@app.get("/health")
async def health():
    return {"message": "OK"}

# Global vars (lazy init)
bot_instance: Optional['DarkBot'] = None
application: Optional[Application] = None

# =========================
# Dark Bot Class (minor fixes: consistent model, added missing self.users_interacted init)
# =========================
class DarkBot:
    def __init__(self) -> None:
        logger.info("=== Dark Bot (Multimodal) Initialization Starting ===")
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.a4f_api_key = os.getenv("A4F_API_KEY")
        if not self.telegram_token or not self.a4f_api_key:
            logger.error("❌ Missing required environment variables.")
            raise ValueError("TELEGRAM_BOT_TOKEN and A4F_API_KEY are required")
        self.client = OpenAI(
            api_key=self.a4f_api_key,
            base_url="https://api.a4f.co/v1",
        )
        # In-memory state
        self.user_memory: dict[int, list[dict]] = {}
        self.group_memory: dict[int, list[dict]] = {}
        self.users_interacted: dict[int, dict] = {}  # Fixed: was missing in one spot
        # Owner info
        self.owner_username = "gothicbatman"
        self.owner_user_id: int | None = None
        logger.info("✅ Dark Bot (Multimodal) initialized successfully")

    # ... (All your memory helpers unchanged)

    def add_to_user_memory(
        self,
        user_id: int,
        user_message: str,
        bot_response: str,
        user_name: str,
        chat_type: str,
        chat_title: str | None = None,
        media_type: str | None = None,
    ) -> None:
        if user_id not in self.user_memory:
            self.user_memory[user_id] = []
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_name": user_name,
            "user_message": user_message,
            "bot_response": bot_response,
            "chat_type": chat_type,
            "chat_title": chat_title or "Private Chat",
            "media_type": media_type,
        }
        self.user_memory[user_id].append(entry)
        self.user_memory[user_id] = self.user_memory[user_id][-15:]

    def add_to_group_memory(
        self,
        chat_id: int,
        user_name: str,
        user_message: str,
        bot_response: str,
        chat_title: str,
        media_type: str | None = None,
    ) -> None:
        if chat_id not in self.group_memory:
            self.group_memory[chat_id] = []
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_name": user_name,
            "user_message": user_message,
            "bot_response": bot_response,
            "chat_title": chat_title,
            "media_type": media_type,
        }
        self.group_memory[chat_id].append(entry)
        self.group_memory[chat_id] = self.group_memory[chat_id][-25:]

    def get_user_memory_context(self, user_id: int, user_name: str) -> str:
        convs = self.user_memory.get(user_id, [])
        if not convs:
            return f"This is my first personal conversation with {user_name}."
        lines = [f"My personal conversation history with {user_name}:"]
        for i, conv in enumerate(convs, start=1):
            chat_location = (
                f"({conv['chat_title']})"
                if conv["chat_type"] != "private"
                else "(Private)"
            )
            media_info = f" [{conv['media_type']}]" if conv.get("media_type") else ""
            user_msg = conv["user_message"]
            bot_msg = conv["bot_response"]
            lines.append(
                f"{i}. {chat_location}{media_info} "
                f"User: {user_msg[:60]}{'...' if len(user_msg) > 60 else ''}"
            )
            lines.append(
                f" My reply: {bot_msg[:60]}{'...' if len(bot_msg) > 60 else ''}"
            )
        return "\n".join(lines)

    def get_group_memory_context(self, chat_id: int, chat_title: str) -> str:
        convs = self.group_memory.get(chat_id, [])
        if not convs:
            return f"This is a new group conversation in {chat_title}."
        lines = [f"Recent group conversation history in {chat_title}:"]
        for i, conv in enumerate(convs, start=1):
            media_info = f" [{conv['media_type']}]" if conv.get("media_type") else ""
            user_msg = conv["user_message"]
            bot_msg = conv["bot_response"]
            lines.append(
                f"{i}. {conv['user_name']}{media_info}: "
                f"{user_msg[:50]}{'...' if len(user_msg) > 50 else ''}"
            )
            lines.append(
                f" My reply: {bot_msg[:50]}{'...' if len(bot_msg) > 50 else ''}"
            )
        return "\n".join(lines)

    # Owner / Creator Helpers (unchanged)
    def is_owner(self, user_id: int, username: str | None = None) -> bool:
        if username and username.lower() == self.owner_username.lower():
            self.owner_user_id = user_id
            return True
        if self.owner_user_id is None:
            return False
        return user_id == self.owner_user_id

    def is_creator_question(self, message: str) -> str | None:
        creator_keywords = [
            "who is your creator",
            "who created you",
            "who made you",
            "your creator",
            "who built you",
            "who designed you",
            "who is your god",
            "your lord",
            "who do you worship",
        ]
        coding_keywords = [
            "who coded you",
            "who programmed you",
            "who wrote you",
            "who developed you",
            "your programmer",
            "your developer",
        ]
        text = message.lower()
        if any(k in text for k in creator_keywords):
            return "creator"
        if any(k in text for k in coding_keywords):
            return "coder"
        return None

    # OpenAI / Multimodal (fixed model consistency)
    async def get_openai_response(
        self,
        prompt: str,
        model: str = "provider-2/gpt-4.1-nano",
        image_data: str | None = None,
    ) -> str:
        try:
            logger.info(f"🔄 Making API call to {model}...")
            if image_data:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ],
                    }
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
            loop = asyncio.get_event_loop()
            def sync_call() -> str:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=10,
                )
                return completion.choices[0].message.content
            response = await loop.run_in_executor(None, sync_call)
            logger.info("✅ API call successful")
            return response
        except Exception as e:
            logger.error(f"❌ Detailed API error: {type(e).__name__}: {e}")
            return "I'm having technical difficulties right now. Give me a moment."

    async def convert_image_to_base64(self, image_bytes: bytes) -> str | None:
        """Convert raw image bytes to optimized base64 JPEG string."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            max_size = 2048
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            if image.mode != "RGB":
                image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Image conversion error: {e}")
            return None

    # All your handlers unchanged (handle_photo, handle_voice, etc. — paste the full ones from your original code here for brevity)
    # ... (start_command, help_command, memory_command, etc. — all the same)

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        logger.error(f"❌ Update {update} caused error {context.error}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # ... (full handler from original — unchanged)

# Lazy init function
async def get_bot_app():
    global bot_instance, application
    if bot_instance is None:
        try:
            bot_instance = DarkBot()
            application = Application.builder().token(bot_instance.telegram_token).build()
            # Add all handlers (as in original)
            application.add_handler(CommandHandler("start", bot_instance.start_command))
            application.add_handler(CommandHandler("help", bot_instance.help_command))
            application.add_handler(CommandHandler("memory", bot_instance.memory_command))
            application.add_handler(CommandHandler("groupmemory", bot_instance.groupmemory_command))
            application.add_handler(CommandHandler("clear", bot_instance.clear_command))
            application.add_handler(CommandHandler("report", bot_instance.report_command))
            application.add_handler(MessageHandler(filters.PHOTO, bot_instance.handle_photo))
            application.add_handler(MessageHandler(filters.VOICE, bot_instance.handle_voice))
            application.add_handler(MessageHandler(filters.Document.ALL, bot_instance.handle_document))
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_instance.handle_message))
            application.add_error_handler(bot_instance.error_handler)
            logger.info("✅ Bot and application initialized")
        except Exception as e:
            logger.error(f"❌ Failed to init bot: {e}")
            raise
    return application

# Webhook endpoint
@app.post("/webhook/{token}")
async def webhook_handler(token: str, request: Request):
    if token != os.getenv("TELEGRAM_BOT_TOKEN"):
        raise HTTPException(status_code=403, detail="Invalid token")
    try:
        data = await request.json()
        app = await get_bot_app()  # Lazy init
        update = Update.de_json(data, app.bot)
        if update:
            await app.process_update(update)
        return {"ok": True}
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal error")

# Local testing fallback
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
