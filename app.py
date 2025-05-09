import os
import re
import logging
from datetime import datetime, timedelta
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError
from slack_sdk import WebClient
from dotenv import load_dotenv
from openai import OpenAI
from dateutil import parser as date_parser

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SlackBot")

app = App(token=os.getenv("SLACK_BOT_TOKEN"))
client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

user_cache = {}

def get_channel_name(channel_id):
    try:
        result = client.conversations_info(channel=channel_id)
        return result["channel"].get("name", channel_id)
    except SlackApiError as e:
        logger.error(f"Slack API error while getting channel name: {e.response['error']}")
        return channel_id

def fetch_recent_messages(channel_id, count=100, days_back=None):
    try:
        latest = datetime.now()
        oldest = latest - timedelta(days=days_back or 1)
        oldest_ts = oldest.timestamp()

        response = client.conversations_history(
            channel=channel_id,
            limit=count,
            oldest=oldest_ts
        )
        return [m for m in response.get("messages", []) if m.get("subtype") != "bot_message"]
    except SlackApiError as e:
        logger.error(f"Error fetching messages: {e.response['error']}")
        return []

def fetch_messages_around_time(channel_id, target_time, minutes=10):
    try:
        delta = timedelta(minutes=minutes)
        oldest_ts = (target_time - delta).timestamp()
        latest_ts = (target_time + delta).timestamp()

        response = client.conversations_history(
            channel=channel_id,
            oldest=oldest_ts,
            latest=latest_ts,
            inclusive=True,
            limit=100
        )
        return [m for m in response.get("messages", []) if m.get("subtype") != "bot_message"]
    except SlackApiError as e:
        logger.error(f"Error fetching messages: {e.response['error']}")
        return []

def summarize_messages(messages):
    cleaned = [f"User <@{m.get('user', 'unknown')}> said: {m.get('text', '')}" for m in messages if 'text' in m]
    if not cleaned:
        return "No content found to summarize."

    prompt = "\n".join(cleaned)
    full_prompt = (
        "Summarize the following Slack conversation. Use clear, brief language suitable for team updates:\n"
        f"{prompt}"
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant summarizing Slack conversations."},
                {"role": "user", "content": full_prompt},
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI summarization error: {e}")
        return "Failed to generate summary."

def index_channel(channel_id):
    channel_name = get_channel_name(channel_id)
    logger.info(f"Started indexing channel {channel_name} ({channel_id})")
    return f"Started indexing channel #{channel_name}."

def extract_requested_timeframe(text):
    if "yesterday" in text:
        return 1
    elif "today" in text:
        return 0
    return None

def extract_channel_id(text):
    match = re.search(r"<#(\w+)(?:\|[^>]+)?>", text)
    return match.group(1) if match else None

def get_user_name(user_id):
    if user_id in user_cache:
        return user_cache[user_id]
    try:
        info = client.users_info(user=user_id)
        name = info["user"].get("real_name") or info["user"].get("name")
        user_cache[user_id] = name
        return name
    except SlackApiError as e:
        logger.error(f"Error fetching user name for {user_id}: {e.response['error']}")
        return user_id

def resolve_user_name(name):
    try:
        users = client.users_list().get("members", [])
        lower_name = name.lower().strip("@")
        for user in users:
            if (
                user.get("name", "").lower() == lower_name or
                user.get("real_name", "").lower() == lower_name or
                user.get("profile", {}).get("display_name", "").lower() == lower_name or
                user.get("id", "") == name
            ):
                return user.get("id")
    except SlackApiError as e:
        logger.error(f"Failed to resolve user name '{name}': {e.response['error']}")
    return None

def extract_datetime(text):
    try:
        return date_parser.parse(text, fuzzy=True)
    except Exception:
        return None

@app.event("app_mention")
def handle_app_mention(body, say):
    event = body.get("event", {})
    text = event.get("text", "")
    lower_text = text.lower()
    channel_id = event.get("channel")
    channel_name = get_channel_name(channel_id)

    logger.info(f"Received mention: {text}")

    if "index this channel" in lower_text or "index channel" in lower_text:
        target_channel_id = extract_channel_id(text) or channel_id
        say(index_channel(target_channel_id))
        return

    if any(kw in lower_text for kw in ["summarize", "recap", "what was discussed", "what happened"]):
        days_back = extract_requested_timeframe(lower_text)
        messages = fetch_recent_messages(channel_id, days_back=days_back)
        say(summarize_messages(messages))
        return

    match = re.search(r"what did (.+?) (say|ask|mention|share|do).*?(\d{1,2}[:\.]\d{2}( ?(am|pm))?( on [^\n\r]+)?)", lower_text)
    if match:
        user_name = match.group(1).strip()
        time_text = match.group(3).strip()
        user_id = resolve_user_name(user_name)

        if not user_id:
            say(f"Sorry, I couldn't match the name '{user_name}' to a Slack user.")
            return

        target_time = extract_datetime(time_text) or datetime.now()
        messages = fetch_messages_around_time(channel_id, target_time)

        hits = [m for m in messages if m.get("user") == user_id and 'text' in m]
        if hits:
            closest = sorted(hits, key=lambda m: abs(float(m["ts"]) - target_time.timestamp()))[0]
            timestamp = datetime.fromtimestamp(float(closest['ts'])).strftime('%I:%M %p')
            say(f"<@{closest['user']}> said at {timestamp}: {closest['text']}")
        else:
            say("No close messages found near that time.")
        return

    if "who is in this channel" in lower_text or "user list" in lower_text:
        try:
            members = client.conversations_members(channel=channel_id)["members"]
            user_names = [get_user_name(uid) for uid in members]
            say("Users in this channel: " + ", ".join(user_names))
        except Exception as e:
            logger.error(f"Error fetching users: {e}")
            say("Couldn't retrieve the user list.")
        return

    say("I'm not sure how to respond. Try things like `summarize this channel`, `index this channel`, `who is in this channel?`, or `what did Alice say at 3:40?`.")

def main():
    logger.info("⚡️ Slack Bot is starting...")
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()

if __name__ == "__main__":
    main()

