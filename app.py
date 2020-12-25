from flask import Flask
from flask_restx import Api

from controllers.dectect_controller import TextDetectController
from detect.model_singleton.crnn_model_singleton import CrnnSingleton

app = Flask(__name__)
api = Api(app, version='1.0', title='Detect API',
    description='A simple Detect API',
)

CrnnSingleton.getModel()

api.add_resource(TextDetectController,'/api/text-detect','/api/text-detect')

if __name__ == '__main__':
    app.run()
