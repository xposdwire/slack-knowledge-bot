import os
import re
import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
from datetime import datetime, timedelta
from slack_sdk import WebClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SlackBot")

# Load environment variables
load_dotenv()

# Initialize Slack app
app = App(token=os.getenv("SLACK_BOT_TOKEN"))
client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

# --- Helpers ---
def get_channel_name(channel_id):
    try:
        result = app.client.conversations_info(channel=channel_id)
        return result["channel"]["name"]
    except SlackApiError as e:
        logger.error(f"Slack API error while getting channel name: {e.response['error']}")
        return None

def index_channel(channel_id):
    try:
        channel_name = get_channel_name(channel_id)
        if not channel_name:
            return f"Couldn't find a channel with ID '{channel_id}'."

        logger.info(f"Indexing channel {channel_name} ({channel_id})")
        return f"Started indexing channel #{channel_name}."
    except Exception as e:
        logger.error(f"Error indexing channel: {e}")
        return f"Failed to index channel due to an error: {str(e)}"

def extract_timestamp_reference(text):
    match = re.search(r'(?:around|at|on)?\s?(\d{1,2}[:.]\d{2})\s?(am|pm)?', text, re.IGNORECASE)
    if match:
        try:
            time_str = match.group(1).replace('.', ':') + (match.group(2) or '')
            parsed_time = datetime.strptime(time_str.strip(), "%I:%M%p")
            return parsed_time.time()
        except Exception as e:
            logger.warning(f"Couldn't parse time reference: {e}")
    return None

def extract_user_reference(text):
    match = re.search(r"what did\s+(@?\w+)", text, re.IGNORECASE)
    if match:
        return match.group(1).lstrip("@")
    return None

def summarize_recent_messages(channel_id, user_filter=None, keyword=None, time_ref=None):
    try:
        now = datetime.now()
        oldest = now - timedelta(days=1)
        result = client.conversations_history(channel=channel_id, oldest=oldest.timestamp())
        messages = result.get("messages", [])

        filtered = []
        for m in messages:
            user_match = not user_filter or m.get("user", "").lower() == user_filter.lower()
            text_match = not keyword or keyword.lower() in m.get("text", "").lower()
            time_match = True
            if time_ref:
                msg_ts = datetime.fromtimestamp(float(m["ts"]))
                diff = abs((datetime.combine(now.date(), time_ref) - msg_ts).total_seconds())
                time_match = diff < 900  # within 15 minutes

            if user_match and text_match and time_match:
                filtered.append(m)

        if not filtered:
            return "I couldn't find any matching messages."

        response_lines = []
        for m in filtered[:5]:
            ts = datetime.fromtimestamp(float(m["ts"])).strftime("%I:%M %p")
            response_lines.append(f"{m.get('user', 'Someone')} said at {ts}: {m.get('text')}")
        return "\n".join(response_lines)

    except Exception as e:
        logger.error(f"Error summarizing messages: {e}")
        return "Sorry, something went wrong while summarizing messages."

# --- Event Handlers ---
@app.event("app_mention")
def handle_app_mention_events(body, say):
    event = body.get("event", {})
    text = event.get("text", "")
    channel_id = event.get("channel")

    logger.info(f"Received mention: {text}")

    lowered = text.lower()
    if "index this channel" in lowered or "index channel" in lowered:
        response = index_channel(channel_id)
        say(response)
    elif "what was discussed" in lowered or "what did" in lowered or "summarize" in lowered:
        time_ref = extract_timestamp_reference(text)
        user_ref = extract_user_reference(text)
        summary = summarize_recent_messages(channel_id, user_filter=user_ref, time_ref=time_ref)
        say(summary)
    elif "who is in this channel" in lowered or "user list" in lowered:
        try:
            members = client.conversations_members(channel=channel_id)["members"]
            users_info = [client.users_info(user=u)["user"]["name"] for u in members]
            say("Users in this channel: " + ", ".join(users_info))
        except Exception as e:
            logger.error(f"Error fetching members: {e}")
            say("Couldn't retrieve member list.")
    else:
        say("I'm not sure how to respond to that. Try saying something like `index this channel`, `what was discussed yesterday?`, or `who is in this channel?`.")

# --- App Runner ---
def main():
    logger.info("⚡️ Slack Bot is starting...")
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()

if __name__ == "__main__":
    main()

