TOKEN = '1414105062:AAGczNbvxlmpm0QV4M7GnkJiI5-fyrmty34'
NGROK_URL = 'https://ttcscn.tk/chatbot'
BASE_TELEGRAM_URL = 'https://api.telegram.org/bot{}'.format(TOKEN)
LOCAL_WEBHOOK_ENDPOINT = '{}/webhook'.format(NGROK_URL)
TELEGRAM_INIT_WEBHOOK_URL = '{}/setWebhook?url={}'.format(BASE_TELEGRAM_URL, LOCAL_WEBHOOK_ENDPOINT)
TELEGRAM_SEND_MESSAGE_URL = BASE_TELEGRAM_URL + '/sendMessage?chat_id={}&text={}'
TELEGRAM_GET_PHOTO_INFO_URL = BASE_TELEGRAM_URL + '/getFile?file_id={}'
TELEGRAM_GET_PHOTO_DATA_URL = 'https://api.telegram.org/file/bot{}'.format(TOKEN) + '/{}'
