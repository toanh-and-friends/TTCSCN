import sys

from flask import request, jsonify
from flask_restx import  Resource

from commons.Type.crnn_mode import CrnnType
from commons.constanst import Constants

from detect.text_detect import TextRecognize
from helpers.file_helpers import create_file_path, save_file, create_folder

import uuid

class TextDetectController(Resource):
    def get(self):
        textRecognize = TextRecognize(
                           None,
                           valid_folder_path=Constants.CRNN_VALID_FOLDER,
                           max_train_files=0,
                           mode=CrnnType.PREDICT,
                           output_test_folder_path=Constants.CRNN_OUPUT_DATA_DIRECTORY,
                           model_file_path=Constants.CRNN_MODEL_PATH)
        data = textRecognize.predict()
        return jsonify(data)

    def post(self):
        try:
            folder_id = str(uuid.uuid4())

            file = request.files['image']
            folder_path = create_folder(Constants.CRNN_OUPUT_DATA_DIRECTORY, folder_id)

            save_status, ex = save_file(folder_path, file.filename, file)

            if(save_status):
                textRecognize = TextRecognize(
                    None,
                    valid_folder_path=folder_path,
                    max_train_files=0,
                    mode=CrnnType.PREDICT,
                    output_test_folder_path=Constants.CRNN_OUPUT_DATA_DIRECTORY,
                    model_file_path=Constants.CRNN_MODEL_PATH)

                data = textRecognize.predict()

                return {
                    "id": folder_id,
                    "data": data
                }
        except Exception as e:
            ex = str(e)

        return {
            "400": "bad request",
            "Exception": ex
        }