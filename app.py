from flask import Flask
from flask_restx import Api

from config import TELEGRAM_INIT_WEBHOOK_URL
from controllers.recogn_controller import TextRecognController
from controllers.telegram_controller import TelegramBotController
from services.bots.telegram_bot import TelegramBot
from services.crnn_recognize.model_singleton.crnn_model_singleton import CrnnSingleton

app = Flask(__name__)
api = Api(app, version='1.0', title='Detect API',
    description='A simple Detect API',
)

CrnnSingleton.getModel()

api.add_resource(TextRecognController,'/api/text-recognize','/api/text-recognize')
api.add_resource(TelegramBotController,'/webhook', '/webhook')

TelegramBot.init_webhook(TELEGRAM_INIT_WEBHOOK_URL)

if __name__ == '__main__':
    app.run()
