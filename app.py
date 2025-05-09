#app.py
import os
import re
import logging
from datetime import datetime, timedelta
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError
from slack_sdk import WebClient
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SlackBot")

# Initialize Slack and OpenAI
app = App(token=os.getenv("SLACK_BOT_TOKEN"))
client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Helper Functions ---
def get_channel_name(channel_id):
    try:
        result = client.conversations_info(channel=channel_id)
        return result["channel"]["name"]
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
        return response.get("messages", [])
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
        response = openai.ChatCompletion.create(
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

def is_requesting_summary(text):
    return any(kw in text for kw in ["summarize", "recap", "what was discussed", "what happened"])

def is_requesting_user_list(text):
    return "who is in this channel" in text or "user list" in text

# --- NLP-based Command Handler ---
@app.event("app_mention")
def handle_app_mention(body, say):
    event = body.get("event", {})
    text = event.get("text", "").lower()
    channel_id = event.get("channel")

    logger.info(f"Received mention: {text}")

    if "index this channel" in text or "index channel" in text:
        target_channel_id = extract_channel_id(text) or channel_id
        say(index_channel(target_channel_id))

    elif is_requesting_summary(text):
        days_back = extract_requested_timeframe(text)
        messages = fetch_recent_messages(channel_id, days_back=days_back)
        say(summarize_messages(messages))

    elif is_requesting_user_list(text):
        try:
            members = client.conversations_members(channel=channel_id)["members"]
            user_names = [client.users_info(user=uid)["user"]["name"] for uid in members]
            say("Users in this channel: " + ", ".join(user_names))
        except Exception as e:
            logger.error(f"Error fetching users: {e}")
            say("Couldn't retrieve the user list.")

    else:
        say("I'm not sure how to respond. Try things like `summarize this channel`, `index this channel`, or `who is in this channel?`.")

# --- Entry Point ---
def main():
    logger.info("⚡️ Slack Bot is starting...")
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()

if __name__ == "__main__":
    main()
