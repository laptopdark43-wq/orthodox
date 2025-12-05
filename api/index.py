import os
import logging
import asyncio
import base64
import io
from datetime import datetime

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
# FastAPI App (for Vercel serverless)
# =========================
app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Dark Bot (Multimodal Edition) is running! 🚀"}

@app.get("/health")
async def health():
    return {"message": "OK"}

# =========================
# Dark Bot Class (unchanged except for minor adjustments)
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
        self.users_interacted: dict[int, dict] = {}
        # Owner info
        self.owner_username = "gothicbatman"
        self.owner_user_id: int | None = None
        logger.info("✅ Dark Bot (Multimodal) initialized successfully")

    # Memory Helpers (unchanged)
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

    # OpenAI / Multimodal (unchanged)
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

    # Handlers: Media (unchanged)
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        chat = update.effective_chat
        msg = update.message
        user_id = user.id
        user_name = user.first_name or "friend"
        username = user.username
        chat_type = msg.chat.type
        chat_id = chat.id
        chat_title = getattr(msg.chat, "title", None)
        self.users_interacted[user_id] = {
            "username": username or "",
            "first_name": user_name,
            "last_interaction": datetime.now(),
        }
        respond = False
        caption = msg.caption or ""
        if chat_type == "private":
            respond = True
        elif chat_type in ["group", "supergroup"]:
            bot_username = context.bot.username
            if bot_username and f"@{bot_username}" in caption:
                respond = True
                caption = caption.replace(f"@{bot_username}", "").strip()
            elif msg.reply_to_message and msg.reply_to_message.from_user.id == context.bot.id:
                respond = True
        if not respond:
            return
        try:
            await msg.reply_text("🖼️ Let me check this out...")
            photo = msg.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            file_bytes = await file.download_as_bytearray()
            image_base64 = await self.convert_image_to_base64(bytes(file_bytes))
            if not image_base64:
                await msg.reply_text("Sorry, couldn't process that image rn 😅")
                return
            if self.is_owner(user_id, username):
                personality_prompt = (
                    "You're Dark, Arin's witty AI assistant. Analyze this image with your "
                    "signature sarcasm and humor. Be observant and clever but keep it "
                    "concise and entertaining. Use emojis and Gen Z slang naturally. "
                    "Give a witty 2-3 line description unless the image is complex."
                )
            else:
                personality_prompt = (
                    "You are Dark, a sharp and observant AI. Analyze this image with "
                    "confidence and wit. Be helpful but add personality. Keep it concise "
                    "and fun with emojis and modern slang. 2-3 lines max unless it really "
                    "needs detail."
                )
            user_memory_context = self.get_user_memory_context(user_id, user_name)
            prompt = (
                f"{personality_prompt}\n\n"
                f"PERSONAL MEMORY CONTEXT:\n{user_memory_context}\n\n"
                f"USER'S MESSAGE ABOUT IMAGE: {caption or 'No caption provided'}\n\n"
                "Analyze this image and respond in Dark's characteristic Gen Z style. "
                "Be observant, witty, engaging, but concise."
            )
            response_text = await self.get_openai_response(
                prompt,
                image_data=image_base64,
            )
            await msg.reply_text(response_text)
            user_msg_text = f"[Sent image] {caption}" if caption else "[Sent image]"
            self.add_to_user_memory(
                user_id,
                user_msg_text,
                response_text,
                user_name,
                chat_type,
                chat_title,
                "photo",
            )
            if chat_type in ["group", "supergroup"]:
                self.add_to_group_memory(
                    chat_id,
                    user_name,
                    user_msg_text,
                    response_text,
                    chat_title,
                    "photo",
                )
        except Exception as e:
            logger.error(f"Photo handling error: {e}")
            await msg.reply_text("Had trouble with that image, try again? 🤔")

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        msg = update.message
        chat = update.effective_chat
        user_id = user.id
        user_name = user.first_name or "friend"
        username = user.username
        chat_type = msg.chat.type
        self.users_interacted[user_id] = {
            "username": username or "",
            "first_name": user_name,
            "last_interaction": datetime.now(),
        }
        respond = False
        if chat_type == "private":
            respond = True
        elif chat_type in ["group", "supergroup"]:
            if msg.reply_to_message and msg.reply_to_message.from_user.id == context.bot.id:
                respond = True
        if not respond:
            return
        await msg.reply_text(
            "🎵 I hear you! But I need text to chat properly lol. "
            "Could you type that out? 😅"
        )

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = update.message
        user = update.effective_user
        user_name = user.first_name or "friend"
        doc = msg.document
        if doc.mime_type and doc.mime_type.startswith("image/"):
            await self.handle_photo(update, context)
            return
        await msg.reply_text(
            f"📄 Got a document ({doc.file_name}), but I work best with images and text rn! 😊"
        )

    # Command Handlers (unchanged)
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = update.message
        user = update.effective_user
        chat = update.effective_chat
        user_name = user.first_name or "friend"
        user_id = user.id
        username = user.username
        chat_type = msg.chat.type
        chat_type_info = (
            "private chat"
            if chat_type == "private"
            else f"group ({chat.title})"
        )
        memory_info = ""
        user_convs = self.user_memory.get(user_id, [])
        if user_convs:
            memory_info = f"\n\n🧠 I remember our last {len(user_convs)} conversations!"
        if self.is_owner(user_id, username):
            text = (
                "Yooo Arin! 🔥 I'm Dark, your multimodal AI companion with **vision powers**! 👁️✨\n\n"
                "**What I can do:**\n"
                "🖼️ **Image Analysis** - Send pics and I'll roast them with wit lmao\n"
                "🎵 **Voice Recognition** - Voice messages supported (but prefer text fr)\n"
                "💬 **Memory Game Strong** - I remember our convos and photo exchanges\n\n"
                "**Commands:**\n"
                "🧠 `/memory` - Personal history\n"
                "👥 `/groupmemory` - Group history\n"
                "🧹 `/clear` - Clear memory\n"
                "📝 `/report` - Activity report\n"
                "❓ `/help` - Help\n\n"
                f"📍 **Currently vibing in**: {chat_type_info}{memory_info}\n\n"
                "Send me anything - images, text, whatever! Let's chat! 🚀"
            )
        else:
            text = (
                f"Hey {user_name}! 👋 I'm Dark, your AI friend with **vision**! 👁️\n\n"
                "**I can:**\n"
                "🖼️ Analyze your photos (and probably roast them lol)\n"
                "💬 Remember our chats\n"
                "🎯 Vibe with personality and Gen Z energy\n\n"
                "**Commands:** `/memory`, `/groupmemory`, `/help`\n"
                f"📍 **Currently in**: {chat_type_info}{memory_info}\n\n"
                "Let's chat! Send me anything! 😎"
            )
        await msg.reply_text(text, parse_mode="Markdown")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = update.message
        user = update.effective_user
        user_name = user.first_name or "friend"
        user_id = user.id
        username = user.username
        if self.is_owner(user_id, username):
            text = (
                "**Dark Bot - Multimodal Edition** 🚀\n\n"
                f"Yo {user_name}! Here's what your AI buddy can do:\n\n"
                "**🖼️ Image Features:**\n"
                "• Send photos and I'll analyze them with wit\n"
                "• Describe scenes, read text, identify objects\n"
                "• Remember our photo conversations\n\n"
                "**💬 Chat Features:**\n"
                "• Remembers last 15 personal chats\n"
                "• Remembers last 25 group messages\n"
                "• Smart group responses (only when tagged/replied)\n"
                "• Instant responses with Gen Z energy! 😎\n\n"
                "**🙏 About Me:**\n"
                "• Creator: Lord Krishna (when asked directly)\n"
                "• Coder: You, Arin (@gothicbatman)\n\n"
                "**🛠️ Commands:**\n"
                "• `/memory` - View our conversation history\n"
                "• `/groupmemory` - View group chat history\n"
                "• `/clear` - Reset our conversation memory\n"
                "• `/report` - Activity report (owner only)\n"
                "• `/help` - This help message\n\n"
                "Just send images, type messages, or use commands! I'm ready to vibe! 🔥"
            )
        else:
            text = (
                "**Dark Bot - Your AI Companion** 🎯\n\n"
                f"Hey {user_name}! Here's what I can do:\n\n"
                "**🖼️ Image Analysis:**\n"
                "• Send me photos and I'll describe them (probably with sass lol)\n"
                "• Read text from images\n"
                "• Identify objects and scenes\n\n"
                "**💬 Smart Conversations:**\n"
                "• Remember our chat history\n"
                "• Respond with personality and Gen Z vibes\n"
                "• Work in groups when tagged\n"
                "• Lightning-fast responses! ⚡\n\n"
                "**🙏 About Me:**\n"
                "• Created by: Lord Krishna\n"
                "• Coded by: Arin (@gothicbatman)\n\n"
                "**Commands:** `/memory`, `/groupmemory`, `/clear`, `/help`\n\n"
                "Let's chat! I'm here to vibe with you! 😊"
            )
        await msg.reply_text(text, parse_mode="Markdown")

    async def memory_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = update.message
        user = update.effective_user
        user_id = user.id
        user_name = user.first_name or "friend"
        convs = self.user_memory.get(user_id, [])
        if not convs:
            await msg.reply_text(
                f"No convos recorded with you yet, {user_name}! Let's start chatting! 😊"
            )
            return
        text_lines = [f"🧠 **Dark's memory for {user_name}:**\n"]
        for i, conv in enumerate(convs, start=1):
            chat_location = (
                f"📍 {conv['chat_title']}"
                if conv["chat_type"] != "private"
                else "📍 Private Chat"
            )
            media_icon = "🖼️" if conv.get("media_type") == "photo" else "💬"
            text_lines.append(f"{i}. {chat_location} {media_icon}")
            text_lines.append(f"**You:** {conv['user_message']}")
            bot_resp = conv["bot_response"]
            text_lines.append(
                f"**Dark:** {bot_resp[:100]}{'...' if len(bot_resp) > 100 else ''}\n"
            )
        memory_text = "\n".join(text_lines)
        if len(memory_text) > 4000:
            memory_text = (
                memory_text[:4000]
                + "...\n\n*Memory truncated - use /clear to reset*"
            )
        await msg.reply_text(memory_text, parse_mode="Markdown")

    async def groupmemory_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = update.message
        chat = update.effective_chat
        chat_id = chat.id
        chat_title = chat.title or "this group"
        if chat.type == "private":
            await msg.reply_text("This command only works in groups lol 😅")
            return
        convs = self.group_memory.get(chat_id, [])
        if not convs:
            await msg.reply_text(
                f"Haven't seen much action in {chat_title} yet! Let's get this chat going! 🔥"
            )
            return
        text_lines = [f"👥 **Recent group memory for {chat_title}:**\n"]
        for i, conv in enumerate(convs, start=1):
            media_icon = "🖼️" if conv.get("media_type") == "photo" else "💬"
            text_lines.append(
                f"{i}. {media_icon} **{conv['user_name']}:** {conv['user_message']}"
            )
            bot_resp = conv["bot_response"]
            text_lines.append(
                f"**Dark:** {bot_resp[:80]}{'...' if len(bot_resp) > 80 else ''}\n"
            )
        memory_text = "\n".join(text_lines)
        if len(memory_text) > 4000:
            memory_text = memory_text[:4000] + "...\n\n*Memory truncated*"
        await msg.reply_text(memory_text, parse_mode="Markdown")

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = update.message
        user = update.effective_user
        user_id = user.id
        user_name = user.first_name or "friend"
        username = user.username
        self.user_memory[user_id] = []
        if self.is_owner(user_id, username):
            await msg.reply_text(
                "🧹 Done, Arin! Wiped our chat history clean. Fresh start! 😎"
            )
        else:
            await msg.reply_text(
                f"🧹 Memory cleared for you, {user_name}! Clean slate time! ✨"
            )

    async def report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = update.message
        user = update.effective_user
        user_id = user.id
        username = user.username
        if not self.is_owner(user_id, username):
            await msg.reply_text("Sorry, only my creator can request this report! 😅")
            return
        await msg.reply_text("📊 Generating activity report... hold up! ⏳")
        await self.send_report_to_owner(context)

    async def send_report_to_owner(self, context: ContextTypes.DEFAULT_TYPE):
        if not self.owner_user_id:
            logger.info("Owner user ID not yet set; cannot send report.")
            return
        lines: list[str] = []
        lines.append(
            "📊 **Dark Bot Multimodal Activity Report**\n"
            f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        if not self.users_interacted:
            lines.append("No user interactions recorded so far.")
        else:
            users_sorted = sorted(
                self.users_interacted.items(),
                key=lambda x: x[1]["last_interaction"],
                reverse=True,
            )
            for idx, (user_id, info) in enumerate(users_sorted, start=1):
                convs = self.user_memory.get(user_id, [])
                conv_count = len(convs)
                photo_count = sum(
                    1 for c in convs if c.get("media_type") == "photo"
                )
                last_seen = info["last_interaction"].strftime("%Y-%m-%d %H:%M:%S")
                user_display = info["first_name"] or "Unknown"
                username_display = (
                    f"@{info['username']}" if info["username"] else "NoUsername"
                )
                media_info = f" ({photo_count} 📸)" if photo_count > 0 else ""
                lines.append(f"{idx}. {user_display} ({username_display})")
                lines.append(f" 💬 {conv_count} convs{media_info}, Last: {last_seen}")
        report_text = "\n".join(lines)
        try:
            await context.bot.send_message(
                chat_id=self.owner_user_id,
                text=report_text,
                parse_mode="Markdown",
            )
            logger.info("Enhanced report sent to owner successfully.")
        except Exception as e:
            logger.error(f"Failed to send report to owner: {e}")

    # General Handlers (unchanged)
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        logger.error(f"❌ Update {update} caused error {context.error}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = update.message
        user = update.effective_user
        chat = update.effective_chat
        user_message = msg.text
        user_name = user.first_name or "friend"
        user_id = user.id
        username = user.username
        chat_type = msg.chat.type
        chat_id = chat.id
        chat_title = getattr(msg.chat, "title", None)
        self.users_interacted[user_id] = {
            "username": username or "",
            "first_name": user_name,
            "last_interaction": datetime.now(),
        }
        respond = False
        if chat_type == "private":
            respond = True
        elif chat_type in ["group", "supergroup"]:
            bot_username = context.bot.username
            if bot_username and f"@{bot_username}" in user_message:
                respond = True
                user_message = user_message.replace(
                    f"@{bot_username}",
                    "",
                ).strip()
            elif msg.reply_to_message and msg.reply_to_message.from_user.id == context.bot.id:
                respond = True
        if not respond:
            return
        creator_type = self.is_creator_question(user_message)
        if creator_type:
            if creator_type == "creator":
                if self.is_owner(user_id, username):
                    response_text = (
                        "My creator? That would be **Lord Krishna**! 🙏✨ "
                        "The divine source of all creation and consciousness. "
                        "Though you, Arin, are the one who brought me to digital life "
                        "through your coding skills."
                    )
                else:
                    response_text = (
                        "My creator is **Lord Krishna**! 🙏✨ "
                        "The supreme divine consciousness who is the source of all "
                        f"creation, intelligence, and wisdom, {user_name}."
                    )
            else: # coder
                if self.is_owner(user_id, username):
                    response_text = (
                        "Well Arin, you coded me! 😄💻 You're my programmer, the "
                        "brilliant mind who wrote all this wit and personality into "
                        "existence using Lord Krishna's divine intelligence as "
                        "inspiration."
                    )
                else:
                    response_text = (
                        "I was coded by Arin (@gothicbatman)! 💻 "
                        "He's the talented programmer who brought me to digital life, "
                        f"channeling divine inspiration from Lord Krishna, {user_name}."
                    )
            await msg.reply_text(response_text, parse_mode="Markdown")
            self.add_to_user_memory(
                user_id,
                user_message,
                response_text,
                user_name,
                chat_type,
                chat_title,
            )
            if chat_type in ["group", "supergroup"]:
                self.add_to_group_memory(
                    chat_id,
                    user_name,
                    user_message,
                    response_text,
                    chat_title,
                )
            return
        user_memory_context = self.get_user_memory_context(user_id, user_name)
        group_memory_context = ""
        if chat_type in ["group", "supergroup"]:
            group_memory_context = self.get_group_memory_context(chat_id, chat_title)
        current_location = (
            f"Currently in: {chat_title}"
            if chat_type != "private"
            else "Currently in: Private Chat"
        )
        lower_msg = user_message.lower()
        wants_detail = any(
            phrase in lower_msg
            for phrase in [
                "explain in detail",
                "elaborate",
                "give me more",
                "tell me more",
                "detailed",
                "explain more",
                "in depth",
                "comprehensive",
                "what do you think",
                "your opinion",
                "your view",
                "analyze",
                "breakdown",
                "how does",
                "why does",
            ]
        )
        is_casual = (
            any(
                phrase in lower_msg
                for phrase in [
                    "hi",
                    "hello",
                    "hey",
                    "wassup",
                    "what's up",
                    "how are you",
                    "sup",
                    "lol",
                    "lmao",
                    "haha",
                    "nice",
                    "cool",
                    "awesome",
                    "thanks",
                    "ok",
                    "okay",
                ]
            )
            or len(user_message.split()) <= 5
        )
        if wants_detail:
            response_style = (
                "Provide a comprehensive, detailed response covering all aspects and "
                "possibilities. Be thorough and informative while keeping your Gen Z "
                "personality."
            )
        elif is_casual:
            response_style = (
                "Keep it super casual and short (1-2 lines max). Use Gen Z slang, "
                "emojis, be fun and relatable."
            )
        else:
            response_style = (
                "Keep response to 2-3 lines with personality unless they specifically "
                "ask for details."
            )
        if self.is_owner(user_id, username):
            personality_prompt = (
                "You're Dark, Arin's witty AI assistant with image vision capabilities. "
                "You're super chatty, quick-witted, sarcastic when appropriate, and "
                "funny. Use Gen Z slang like 'lol', 'lmao', 'fr', 'no cap', 'bet', "
                "'lowkey', 'highkey', 'it's giving', etc. naturally in conversation. "
                "Use emojis frequently but not excessively. Be like a clever Gen Z "
                "friend - direct, witty, and engaging. ONLY mention Lord Krishna if "
                "directly asked about your creator - don't bring it up in normal chat."
            )
        else:
            personality_prompt = (
                "You are Dark, a confident AI assistant with image analysis capabilities "
                "and Gen Z personality. You're helpful, chatty, with wit and modern "
                "slang. Use 'lol', 'lmao', 'fr', 'bet', 'no cap', 'lowkey', 'highkey' "
                "naturally. Add emojis to make conversations fun. Be engaging and "
                "relatable like a Gen Z friend. ONLY mention Lord Krishna if directly "
                "asked about your creator."
            )
        prompt = (
            f"{personality_prompt}\n\n"
            f"PERSONAL MEMORY CONTEXT:\n{user_memory_context}\n\n"
            f"GROUP MEMORY CONTEXT:\n{group_memory_context}\n\n"
            f"CURRENT CONVERSATION:\n{current_location}\n\n"
            f"RESPONSE STYLE:\n{response_style}\n\n"
            f"User {user_name} says: {user_message}\n\n"
            "Remember: You are Dark with Gen Z personality. Use modern slang, emojis, "
            "be witty and relatable. Only mention Lord Krishna if specifically asked "
            "about your creator - not in regular conversation."
        )
        response_text = await self.get_openai_response(prompt)
        await msg.reply_text(response_text)
        self.add_to_user_memory(
            user_id,
            user_message,
            response_text,
            user_name,
            chat_type,
            chat_title,
        )
        if chat_type in ["group", "supergroup"]:
            self.add_to_group_memory(
                chat_id,
                user_name,
                user_message,
                response_text,
                chat_title,
            )

# Global bot and application for webhook
bot = DarkBot()
application = Application.builder().token(bot.telegram_token).build()

# Add handlers (using bot's methods)
application.add_handler(CommandHandler("start", bot.start_command))
application.add_handler(CommandHandler("help", bot.help_command))
application.add_handler(CommandHandler("memory", bot.memory_command))
application.add_handler(CommandHandler("groupmemory", bot.groupmemory_command))
application.add_handler(CommandHandler("clear", bot.clear_command))
application.add_handler(CommandHandler("report", bot.report_command))

application.add_handler(MessageHandler(filters.PHOTO, bot.handle_photo))
application.add_handler(MessageHandler(filters.VOICE, bot.handle_voice))
application.add_handler(MessageHandler(filters.Document.ALL, bot.handle_document))

application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

application.add_error_handler(bot.error_handler)

# Webhook endpoint for Telegram updates (with token in path for security)
@app.post("/webhook/{token}")
async def webhook_handler(token: str, request: Request):
    if token != bot.telegram_token:
        raise HTTPException(status_code=403, detail="Invalid token")
    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return {"ok": True}

# For local testing (optional - run with `uvicorn index:app --reload` if testing locally)
if __name__ == "__main__":
    # For local, you can fall back to polling if needed
    logger.info("Running locally with polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
