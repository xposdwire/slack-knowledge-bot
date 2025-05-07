# app.py
import os
import re
import io
import json
import base64
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

import openai
from dateutil import parser as dt_parser

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SlackBot")

DATA_DIR = Path("channel_data")
MESSAGE_HISTORY_FILE = DATA_DIR / "message_history.pkl"

app = App(token=os.getenv("SLACK_BOT_TOKEN"))
openai.api_key = os.getenv("OPENAI_API_KEY")

class SlackKnowledgeBot:
    def __init__(self):
        self.channel_messages = {}
        self.document_cache = {}
        self.user_info_cache = {}
        self.last_indexed = {}
        DATA_DIR.mkdir(exist_ok=True)
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
            target_dt = dt_parser.parse(time_str, fuzzy=True, default=datetime.now())
            closest_msg = None
            smallest_diff = timedelta(minutes=10)

            for msg in self.channel_messages.get(channel_id, []):
                msg_dt = datetime.fromtimestamp(float(msg["ts"]))
                time_diff = abs(target_dt - msg_dt)
                if time_diff < smallest_diff:
                    closest_msg = msg
                    smallest_diff = time_diff

            if closest_msg:
                ts_str = datetime.fromtimestamp(float(closest_msg['ts'])).strftime('%b %d %I:%M %p')
                return f"Closest match at {ts_str}: {closest_msg['user']}: {closest_msg['text']}"
            return f"No close messages found near '{time_str}'."
        except Exception as e:
            logger.error(f"Natural language time parse error: {e}")
            return "I had trouble understanding the time. Try something like 'yesterday at 3 PM' or 'May 5 3:40 PM'."

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
    elif re.search(r"what did (\w+) ask (?:at|on|around|about)? (.+)", query):
        time_str = re.findall(r"ask (?:at|on|around|about)? (.+)", query)[0]
        response = app.knowledge_bot.get_message_by_time(channel_id, time_str)
    else:
        response = app.knowledge_bot._generate_llm_answer(query)

    say(response)

if __name__ == "__main__":
    app.knowledge_bot = SlackKnowledgeBot()
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    handler.start()