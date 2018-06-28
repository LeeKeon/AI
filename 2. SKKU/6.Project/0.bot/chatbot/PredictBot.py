# -*- coding: utf-8 -*-

import json
import websockets
import logging
import requests
import asyncio
from chatbot.Predict import Predict


logger = logging.getLogger('websockets')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class PredictBot:
    def __init__(self, token):
        self.token = token
        self.websocket = None
        self.id = None
        self.pred = Predict('HelloWorld')

    def connect_rtm(self):
        response = requests.post('https://slack.com/api/rtm.start', data={'token': self.token}).json()
        self.id = response.get('self').get('id')

        if response.get('ok') is True and response.get('url') is not None:
            return response.get('url')
        else:
            raise Exception('Slack RTM API Conection is failed..')

    @asyncio.coroutine
    def listen_rtm(self):
        try:
            self.websocket = yield from websockets.connect(self.connect_rtm())
            while True:
                recv_msg = yield from self.websocket.recv()
                msg_json = json.loads(recv_msg)
                channel = msg_json.get('channel')
                user = msg_json.get('user')
                if self.preprocess(msg_json):
                    print(msg_json)
                    messages = self.parse_message(msg_json.get('text'))
                    channel = msg_json.get('channel')
                    user = msg_json.get('user')

                    if len(messages) < 2:
                        yield from self.websocket.send(json.dumps({"id":1, "type": "message", "channel":channel, "text": '<@{0}> {1}'.format(user, 'Input Command')}))
                    else:
                        print('================== from Predict',self.pred.result())
                        yield from self.websocket.send(json.dumps({"id":1, "type": "message", "channel":channel, "text": '<@{0}> {1}'.format(user, messages[1])}))
        except Exception as e:
            print(e)

    def preprocess(self, msg_json):
        print(msg_json)
        if msg_json.get('type') == 'hello':
            print('Slack Bot Connect Success!')
            return False
        elif msg_json.get('type') == 'message' and msg_json.get('text') is not None:
            if msg_json.get('text').startswith('<@{0}>'.format(self.id)):
                 return True
        else:
            return False

    def parse_message(self, message_text):
        message_text = str(message_text)
        print(message_text)
        messages = message_text.split(' ')

        return messages
   
    def send_messege(self, channel, user, message):
       try:
            yield from self.websocket.send(json.dumps({"id":1, "type": "message", "channel":channel, "text": '<@{0}> {1}'.format(user, message)}))
       except Exception as e:
            print(e)
       return True


    # Load whole weights
    def restore(self, sess):
        print(' - Restoring variables...')
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, "models/rnn")
        print(' * model restored ')
