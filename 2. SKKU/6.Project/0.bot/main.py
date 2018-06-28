import secrets
import asyncio
from chatbot.PredictBot import PredictBot

if __name__ == '__main__':
    print(secrets.SLACK_API_TOKEN)
    sb = PredictBot(secrets.SLACK_API_TOKEN)
    asyncio.get_event_loop().run_until_complete(sb.listen_rtm())
