import os

import requests
from flask import json, app
import logging.config
import logging
from config import TELEGRAM_SEND_MESSAGE_URL, TELEGRAM_GET_PHOTO_INFO_URL, TELEGRAM_GET_PHOTO_DATA_URL
import uuid

logger = logging.getLogger("root")

class TelegramBot:
    def __init__(self):
        self.chat_id = None
        self.text = None
        self.first_name = None
        self.last_name = None
        self.incoming_message_photo = None
        self.incoming_message_text = None
        logger.info('Init bot')

    def parse_webhook_data(self, data):
        message = data['message']

        self.chat_id = message['chat']['id']
        self.first_name = message['from']['first_name']
        self.last_name = message['from']['last_name']

        if 'text' in message:
            self.incoming_message_text = message['text'].lower()
        elif 'photo' in message:
            self.incoming_message_photo = message['photo'][0]

    def action(self):
        success = None
        if self.incoming_message_photo is None:
            if self.incoming_message_text == '/start':
                self.outgoing_message_text = 'Hello {} {} 👋! I am text detect bot. ' \
                                             '\n Please post a image to text box ' \
                                             '\n\n\b And other option: ' \
                                             '\n    /start: to start converstion. ' \
                                             '\n    /help: get the help' \
                                             '\n\nThanks for use my bot'.format(self.first_name, self.last_name)
                success = self.send_message()

            if self.incoming_message_text == '/help':
                self.outgoing_message_text = 'Please post a image to text box. And wait few minutes, I will talk you something in image. '
                success = self.send_message()

            if self.incoming_message_text == '/rad':
                self.outgoing_message_text = '🤙'
                success = self.send_message()
        else:
            success = self.get_image()

        logger.info('%s failed to log in', success)
        return success


    def send_message(self):
        res = requests.get(TELEGRAM_SEND_MESSAGE_URL.format(self.chat_id, self.outgoing_message_text))
        return True if res.status_code == 200 else False

    def get_image(self):
        id = str(uuid.uuid4())
        file_id = str(self.incoming_message_photo['file_id'])
        img_info_res = requests.get(TELEGRAM_GET_PHOTO_INFO_URL.format(file_id))
        file_info =  json.loads(img_info_res.content)

        file_url = TELEGRAM_GET_PHOTO_DATA_URL.format(file_info['result']['file_path'])
        file_data_res = requests.get(file_url)

        file_name = './files/google_logo_'+id+'.png'

        with open(file_name, 'wb') as f:
            f.write(file_data_res.content)
            print(file_name)

        return True if file_data_res.status_code == 200 else False

    @staticmethod
    def init_webhook(url):
        requests.get(url)


