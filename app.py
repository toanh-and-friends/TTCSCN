import logging
import os

from flask import Flask, request, jsonify

from telegram_bot import TelegramBot
from config import TELEGRAM_INIT_WEBHOOK_URL

app = Flask(__name__)
TelegramBot.init_webhook(TELEGRAM_INIT_WEBHOOK_URL)

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
logger = logging.getLogger("root")
logging.config.dictConfig(dict_config)


@app.route('/webhook', methods=['POST'])
def index():
    logger.info('webhook')

    req = request.get_json()
    bot = TelegramBot()
    bot.parse_webhook_data(req)
    success = bot.action()
    return jsonify(success=success) # TODO: Success should reflect the success of the reply

if __name__ == '__main__':
    app.run(port=8081, host='0.0.0.0', ssl_context=('cert.pem', 'privkey.pem'))
