from flask import Flask, request, jsonify
from services.bots.telegram_bot import TelegramBot
from flask_restx import Resource

class TelegramBotController(Resource):
    def get(self):
        return {
            "guide": "connect to telegram wwith post method"
        }

    def post(self):
        req = request.get_json()
        bot = TelegramBot()
        bot.parse_webhook_data(req)
        success = bot.action()
        return jsonify(success=success)
