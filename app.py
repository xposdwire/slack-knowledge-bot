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
from difflib import get_close_matches
from slack_bolt.context.say import Say

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SlackBot")

DATA_DIR = Path("channel_data")
MESSAGE_HISTORY_FILE = DATA_DIR / "message_history.pkl"
USER_CACHE_FILE = DATA_DIR / "user_name_cache.pkl"

app = App(token=os.getenv("SLACK_BOT_TOKEN"))
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SlackKnowledgeBot:
    def __init__(self):
        self.channel_messages = {}
        self.document_cache = {}
        self.user_info_cache = {}
        self.last_indexed = {}
        self.name_to_id = {}
        self.channel_name_map = {}
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

        if USER_CACHE_FILE.exists():
            try:
                with open(USER_CACHE_FILE, "rb") as f:
                    self.name_to_id = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load name cache: {e}")

    def _save_data(self):
        try:
            with open(MESSAGE_HISTORY_FILE, "wb") as f:
                pickle.dump({
                    "channel_messages": self.channel_messages,
                    "document_cache": self.document_cache,
                    "user_info_cache": self.user_info_cache,
                    "last_indexed": self.last_indexed
                }, f)
            with open(USER_CACHE_FILE, "wb") as f:
                pickle.dump(self.name_to_id, f)
        except Exception as e:
            logger.error(f"Failed to save data: {e}")

    def _get_user_name(self, user_id):
        if user_id not in self.user_info_cache:
            try:
                info = app.client.users_info(user=user_id)
                profile = info["user"].get("profile", {})
                fields = [
                    info["user"].get("real_name"),
                    profile.get("display_name"),
                    info["user"].get("name")
                ]
                name = next((n for n in fields if n), user_id)
                self.user_info_cache[user_id] = name
                for alias in filter(None, map(str.lower, fields)):
                    self.name_to_id[alias] = user_id
            except Exception as e:
                logger.error(f"Error fetching user info for {user_id}: {e}")
                return user_id
        return self.user_info_cache[user_id]

    def _resolve_user_query(self, query_name):
        if not self.name_to_id:
            for ch_msgs in self.channel_messages.values():
                for msg in ch_msgs:
                    uid = msg.get("user")
                    if uid:
                        self._get_user_name(uid)
        names = list(self.name_to_id.keys())
        best = get_close_matches(query_name.lower(), names, n=1, cutoff=0.6)
        if best:
            return self.name_to_id[best[0]]
        return None

    def _resolve_channel_name(self, name):
        try:
            result = app.client.conversations_list(types="public_channel,private_channel")
            self.channel_name_map = {ch["name"]: ch["id"] for ch in result["channels"]}
            return self.channel_name_map.get(name.strip("#"))
        except Exception as e:
            logger.error(f"Channel resolve failed: {e}")
            return None

    def list_users_in_channel(self, channel_id):
        user_ids = {msg["user"] for msg in self.channel_messages.get(channel_id, []) if "user" in msg}
        users = sorted(self._get_user_name(uid) for uid in user_ids if uid)
        return "Here are the users I've seen in this channel:\n" + "\n".join(f"- {u}" for u in users) if users else "I haven't indexed any users in this channel yet."

    def index_channel(self, channel_id, days_back=7):
        try:
            oldest_ts = (datetime.now() - timedelta(days=days_back)).timestamp()
            all_msgs = []
            cursor = None
            while True:
                result = app.client.conversations_history(
                    channel=channel_id,
                    limit=100,
                    cursor=cursor,
                    oldest=str(oldest_ts)
                )
                messages = result.get("messages", [])
                all_msgs.extend([m for m in messages if "text" in m])
                cursor = result.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break
            for msg in all_msgs:
                msg.setdefault("user", "unknown")
            self.channel_messages[channel_id] = all_msgs
            self.last_indexed[channel_id] = datetime.now().timestamp()
            logger.info(f"Indexed {len(all_msgs)} messages from channel {channel_id}")
            self._save_data()
        except SlackApiError as e:
            logger.error(f"Failed to index {channel_id}: {e}")

    def auto_index_all_channels(self):
        try:
            result = app.client.conversations_list(types="public_channel,private_channel")
            for ch in result.get("channels", []):
                if ch.get("is_member"):
                    self.index_channel(ch["id"], days_back=7)
                else:
                    logger.info(f"Skipping channel {ch['name']} - bot not a member.")
        except Exception as e:
            logger.error(f"Auto index error: {e}")

    def find_messages_by_keyword_and_time(self, channel_id, keyword, time_window):
        try:
            keyword = keyword.lower()
            start = dt_parser.parse(time_window, fuzzy=True, default=datetime.now())
            end = start + timedelta(days=1)
            matches = [
                m for m in self.channel_messages.get(channel_id, [])
                if keyword in m.get("text", "").lower()
                and start.timestamp() <= float(m["ts"]) <= end.timestamp()
            ]
            if not matches:
                return f"No messages found about '{keyword}' on {start.strftime('%Y-%m-%d')}."
            return "\n".join(f"- {datetime.fromtimestamp(float(m['ts'])).strftime('%b %d %I:%M %p')}: <@{m['user']}> {m['text']}" for m in matches[:5])
        except Exception as e:
            logger.error(f"Keyword+time search error: {e}")
            return "Failed to search with time window."

    def _generate_llm_answer(self, query):
        try:
            system_prompt = "You are a helpful assistant in Slack. Use your knowledge and sound like a team member."
            response = openai_client.chat.completions.create(
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

@app.command("/refresh")
def handle_refresh(ack, body, say: Say):
    ack()
    channel_id = body['channel_id']
    say("Re-indexing messages in this channel... this may take a moment.")
    app.knowledge_bot.index_channel(channel_id, days_back=7)
    say("Done! Channel re-indexed.")

@app.event("app_mention")
def handle_app_mention(event, say):
    channel_id = event["channel"]
    text = event["text"]
    user = event.get("user")
    if not hasattr(app, "knowledge_bot"):
        app.knowledge_bot = SlackKnowledgeBot()
    query = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
    logger.info(f"Received mention from {user}: {query}")

    if m := re.search(r"index channel #?(\S+)", query, re.I):
        ch_name = m.group(1)
        ch_id = app.knowledge_bot._resolve_channel_name(ch_name)
        if ch_id:
            app.knowledge_bot.index_channel(ch_id)
            say(f"Channel #{ch_name} has been indexed.")
        else:
            say("Couldn't find that channel.")
    elif m := re.search(r"what was discussed about (.+?) (yesterday|today|last week|on .+?)", query, re.I):
        keyword, time_phrase = m.groups()
        response = app.knowledge_bot.find_messages_by_keyword_and_time(channel_id, keyword.strip(), time_phrase)
        say(response)
    else:
        say(app.knowledge_bot._generate_llm_answer(query))

if __name__ == "__main__":
    app.knowledge_bot = SlackKnowledgeBot()
    app.knowledge_bot.auto_index_all_channels()
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    handler.start()

