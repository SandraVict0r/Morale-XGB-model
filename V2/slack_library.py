import os
from datetime import datetime
import slack_sdk

def post_message_to_slack( text, blocks=None):


    SLACK_TOKEN = "xoxb-4842891246323-4828903932647-TskVDKcbnML4OOHUTQYeJ8l9"
    CHANNEL = '#general' # Insert the name of channel here
    USERNAME = 'Deepnote Bot'
    ICON = 'https://avatars.githubusercontent.com/u/45339858?s=280&v=4' # This is a URL pointing to the icon of your bot.
    today = datetime.today().strftime('%B %d, %Y at %H:%M UTC')
    MESSAGE_TEXT = f'Experiment status: Success!\nLast run: {today}' # This is a sample message. Replace with your own text.

    try:
        print('Sending message...')
        # Connect to the Slack client via token
        client = slack_sdk.WebClient(token=SLACK_TOKEN)

        # Post the Slack message
        request = client.chat_postMessage(
            text=text + MESSAGE_TEXT,
            channel=CHANNEL,
            blocks=blocks,
            icon_url=ICON,
            username=USERNAME
        )
        print('Message was sent successfully.')
        return request

        # Print error
    except Exception as e:
        print(f'Error: {e}')