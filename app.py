import logging

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

#log config
dict_config = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)d} %(levelname)s - %(message)s',
        }
    },
    'handlers': {'default': {
        'level': 'DEBUG',
        'formatter': 'default',
        'class': 'logging.handlers.RotatingFileHandler',
        'filename': "test.log",
        'maxBytes': 5000000,
        'backupCount': 10
    },
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'default',
        },
    },
    'loggers': {
        'myapp': {
            'handlers': ["default"],
            'level': 'DEBUG',
        },
    },
    'root': {
        'handlers': ["console"],
        'level': 'DEBUG',
    },
}

print(__name__)
logging.config.dictConfig(dict_config)


#model config
CrnnSingleton.getModel()

api.add_resource(TextRecognController,'/api/text-recognize','/api/text-recognize')
api.add_resource(TelegramBotController,'/webhook', '/webhook')

TelegramBot.init_webhook(TELEGRAM_INIT_WEBHOOK_URL)

if __name__ == '__main__':
    app.run( host='0.0.0.0', port=8081, ssl_context='adhoc')
