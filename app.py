# app.py
# Full version: Slack bot without OCR/image features

import os
import re
import io
import base64
import tempfile
import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

import openai
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SlackBot")

load_dotenv()

DATA_DIR = Path("channel_data")
MESSAGE_HISTORY_FILE = DATA_DIR / "message_history.pkl"
DOCUMENT_CACHE_DIR = DATA_DIR / "documents"

MAX_DOCUMENT_SIZE = 50 * 1024 * 1024
ALLOWED_DOCUMENT_EXTENSIONS = ['.pdf', '.doc', '.docx', '.txt']

app = App(token=os.getenv("SLACK_BOT_TOKEN"))
openai.api_key = os.getenv("OPENAI_API_KEY")

class DocumentProcessor:
    @staticmethod
    def process_pdf(pdf_data):
        import fitz
        try:
            with fitz.open(stream=pdf_data, filetype="pdf") as doc:
                return "".join(p.get_text() for p in doc).strip()
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return None

    @staticmethod
    def generate_pdf(content, title="Generated Document"):
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name="CustomTitle", parent=styles["Title"], fontSize=16, spaceAfter=12))
            elements = [Paragraph(title, styles['CustomTitle']), Spacer(1, 12)]
            for p in content.split("\n\n"):
                if p.strip():
                    elements.append(Paragraph(p.strip(), styles["Normal"]))
                    elements.append(Spacer(1, 6))
            doc.build(elements)
            buffer.seek(0)
            return buffer.read()
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return None

    @staticmethod
    def generate_chart(data: dict, title: str = "Chart", chart_type: str = "bar"):
        try:
            plt.figure(figsize=(10, 6))
            if chart_type == "bar":
                plt.bar(data.keys(), data.values())
            elif chart_type == "line":
                plt.plot(list(data.keys()), list(data.values()))
            elif chart_type == "pie":
                plt.pie(data.values(), labels=data.keys(), autopct="%1.1f%%")
            plt.title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            plt.close()
            buffer.seek(0)
            return buffer.read()
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return None

class SlackKnowledgeBot:
    def __init__(self):
        self.channel_messages = {}
        self.document_cache = {}
        self.user_info_cache = {}
        self.last_indexed = {}
        DATA_DIR.mkdir(exist_ok=True)
        DOCUMENT_CACHE_DIR.mkdir(exist_ok=True)
        self._load_data()

    def _load_data(self):
        if MESSAGE_HISTORY_FILE.exists():
            try:
                with open(MESSAGE_HISTORY_FILE, "rb") as f:
                    data = pickle.load(f)
                    self.channel_messages = data.get("channel_messages", {})
                    self.document_cache = data.get("document_cache", {})
                    self.user_info_cache = data.get("user_info_cache", {})
                    self.last_indexed = data.get("last_indexed", {})
            except Exception as e:
                logger.error(f"Failed to load message history: {e}")

    def _save_data(self):
        try:
            with open(MESSAGE_HISTORY_FILE, "wb") as f:
                pickle.dump({
                    "channel_messages": self.channel_messages,
                    "document_cache": self.document_cache,
                    "user_info_cache": self.user_info_cache,
                    "last_indexed": self.last_indexed
                }, f)
        except Exception as e:
            logger.error(f"Failed to save message history: {e}")

    def _download_file(self, url):
        try:
            import requests
            headers = {"Authorization": f"Bearer {os.getenv('SLACK_BOT_TOKEN')}"}
            resp = requests.get(url, headers=headers)
            return resp.content if resp.status_code == 200 else None
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None

    def _process_file(self, file_info):
        if file_info.get("size", 0) > MAX_DOCUMENT_SIZE:
            return None
        file_id = file_info.get("id")
        if file_id in self.document_cache:
            return self.document_cache[file_id]

        name = file_info.get("name")
        url = file_info.get("url_private")
        ext = Path(name).suffix.lower()
        content = self._download_file(url)
        text = None
        if ext == ".pdf":
            text = DocumentProcessor.process_pdf(content)
        elif ext == ".txt":
            text = content.decode("utf-8", errors="replace")

        result = {
            "id": file_id,
            "name": name,
            "type": file_info.get("filetype"),
            "size": file_info.get("size"),
            "text": text
        }
        self.document_cache[file_id] = result
        return result

    def _generate_llm_answer(self, query):
        try:
            system_prompt = "You are a helpful assistant in Slack. Answer questions clearly and helpfully."
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=600
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM response error: {e}")
            return "I'm sorry, I couldn't process that request."

    def get_channel_info(self, channel_id):
        messages = self.channel_messages.get(channel_id, [])
        if not messages:
            return "I haven't indexed this channel yet."
        summary = f"This channel has {len(messages)} messages indexed. Last updated on {datetime.fromtimestamp(self.last_indexed.get(channel_id, 0)).strftime('%Y-%m-%d')}"
        return summary

    def get_message_by_time(self, channel_id, time_str):
        try:
            target_time = datetime.strptime(time_str, "%I:%M %p").time()
            for msg in self.channel_messages.get(channel_id, []):
                msg_time = datetime.fromtimestamp(float(msg["ts"])).time()
                if msg_time.hour == target_time.hour and msg_time.minute == target_time.minute:
                    return f"{msg['user']}: {msg['text']}"
            return f"I couldn't find a message at {time_str}."
        except Exception as e:
            logger.error(f"Time parse error: {e}")
            return "I couldn't interpret the time format. Use something like 3:40 PM."

@app.event("app_mention")
def handle_app_mention(event, say):
    channel_id = event["channel"]
    text = event["text"]
    user = event.get("user")
    if not hasattr(app, "knowledge_bot"):
        app.knowledge_bot = SlackKnowledgeBot()
    query = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
    logger.info(f"Received mention from {user}: {query}")

    if re.search(r"what can you tell me about this channel", query, re.I):
        response = app.knowledge_bot.get_channel_info(channel_id)
    elif re.search(r"what did (\w+) ask at ([0-9: ]+[apAP][mM])", query):
        time_str = re.findall(r"at ([0-9: ]+[apAP][mM])", query)[0]
        response = app.knowledge_bot.get_message_by_time(channel_id, time_str)
    else:
        response = app.knowledge_bot._generate_llm_answer(query)

    say(response)

@app.event("file_shared")
def handle_file_shared(event):
    file_id = event.get("file_id")
    logger.info(f"File shared with ID: {file_id}")
    try:
        file_info = app.client.files_info(file=file_id)["file"]
        app.knowledge_bot._process_file(file_info)
        logger.info(f"File processed: {file_info.get('name')}")
    except Exception as e:
        logger.error(f"Error processing shared file: {e}")

@app.event("member_joined_channel")
def handle_member_joined_channel_events(body, logger, say):
    logger.info(f"Bot noticed a member joined: {body}")
    user_id = body["event"].get("user")
    channel_id = body["event"].get("channel")
    if user_id and channel_id:
        say(channel=channel_id, text=f"👋 Welcome <@{user_id}>! Let me know if you need help or want me to index this channel.")

# === Background Auto Indexing ===
def start_auto_indexing():
    import threading

    def update_all_channels():
        logger.info("Running periodic auto-index update...")
        try:
            response = app.client.conversations_list(types="public_channel,private_channel")
            for channel in response["channels"]:
                if not channel.get("is_member", False):
                    logger.info(f"Skipping channel {channel['name']} - bot not a member.")
                    continue
                ch_id = channel["id"]
                now = datetime.now().timestamp()
                last_ts = app.knowledge_bot.last_indexed.get(ch_id, now - 3600)
                logger.info(f"Auto-updating channel {ch_id} since {last_ts}")
                messages = app.client.conversations_history(
                    channel=ch_id,
                    limit=100,
                    oldest=str(last_ts)
                )["messages"]
                existing_ts = {msg["ts"] for msg in app.knowledge_bot.channel_messages.get(ch_id, [])}
                new_msgs = [
                    {"text": m["text"], "ts": m["ts"], "user": m.get("user", "unknown")}
                    for m in messages if m.get("text") and m["ts"] not in existing_ts
                ]
                if new_msgs:
                    app.knowledge_bot.channel_messages.setdefault(ch_id, []).extend(new_msgs)
                    app.knowledge_bot.last_indexed[ch_id] = now
                    logger.info(f"Indexed {len(new_msgs)} new messages for {ch_id}")
            app.knowledge_bot._save_data()
        except Exception as e:
            logger.error(f"Auto indexing failed: {e}")
        finally:
            threading.Timer(900, update_all_channels).start()

    update_all_channels()

if __name__ == "__main__":
    app.knowledge_bot = SlackKnowledgeBot()
    start_auto_indexing()
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    handler.start()
