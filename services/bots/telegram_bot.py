import os

import requests
from flask import json

from commons.Type.crnn_mode import CrnnType
from commons.constanst import Constants
from config import TELEGRAM_SEND_MESSAGE_URL, TELEGRAM_GET_PHOTO_INFO_URL, TELEGRAM_GET_PHOTO_DATA_URL
import uuid

from helpers.file_helpers import create_folder, create_file_name, create_file_path
from services.crnn_recognize.text_detect import TextRecognize
from services.east_detect.cut_image import ImagesData
from services.east_detect.new_predict import TextDetection


class TelegramBot:
    def __init__(self):
        self.chat_id = None
        self.text = None
        self.first_name = None
        self.last_name = None
        self.incoming_message_photo = None
        self.incoming_message_text = None

    def parse_webhook_data(self, data):
        message = data['message']

        self.chat_id = message['chat']['id']
        self.first_name = message['from']['first_name']
        self.last_name = message['from']['last_name']

        if 'text' in message:
            self.incoming_message_text = message['text'].lower()
        elif 'photo' in message:
            self.incoming_message_photo = message['photo']

    def action(self):
        success = None
        if self.incoming_message_photo is None:
            if self.incoming_message_text == '/help':
                self.outgoing_message_text = 'Please post a image to text box. And wait few minutes, I will talk you something in image. '
                success = self.send_message()

            elif self.incoming_message_text == '/rad':
                self.outgoing_message_text = '🤙'
                success = self.send_message()

            else:
                self.outgoing_message_text = 'Hello {} {} 👋! I am text crnn_recognize bot. ' \
                                             '\n Please post a image to text box ' \
                                             '\n\n\b And other option: ' \
                                             '\n    /start: to start converstion. ' \
                                             '\n    /help: get the help' \
                                             '\n\nThanks for use my bot'.format(self.first_name, self.last_name)
                success = self.send_message()

        else:
            data = self.__detect_image()
            self.outgoing_message_text = data
            success = self.send_message()

        return success



    def send_message(self):
        res = requests.get(TELEGRAM_SEND_MESSAGE_URL.format(self.chat_id, self.outgoing_message_text))
        return True if res.status_code == 200 else False

    def __detect_image(self):
        ex = ''

        try:
            status, folder_path, ex, file_path = self.__download_image()
            folder_detect_path = create_folder(Constants.CRNN_OUPUT_DATA_DIRECTORY, self.chat_id)
            foler_crop_img =  create_folder(Constants.CROP_OUTPUT_IMG_DIRECTORY, self.chat_id)

            td = TextDetection(test_data_path=folder_path,
                               model_path=Constants.EAST_MODEL_DIRECTORY,
                               test_data_output_path=folder_detect_path,
                               gpu_num='1')
            detected_file, detect_img = td.predict()

            ig = ImagesData(output_folder_path=foler_crop_img,
                            image_file_path=detect_img,
                            data_image_file_path=detected_file)
            ig.detect_baselines_crop_images()

            if status:
                textRecognize = TextRecognize(
                    None,
                    valid_folder_path=foler_crop_img,
                    max_train_files=0,
                    mode=CrnnType.PREDICT,
                    output_test_folder_path=Constants.CRNN_OUPUT_DATA_DIRECTORY,
                    model_file_path=Constants.CRNN_MODEL_PATH
                )

                predict_result = textRecognize.predict()

                return predict_result
        except Exception as e:
            ex = str(e)
            return ex

    def __values_dir_to_string(self, dict):
        return "".join(dict.predict_texts())

    def __download_image(self):
        file_datas = self.__get_img_resp()
        status, folder_path, ex, file_path = self.__save_images(file_datas)

        return status, folder_path, ex, file_path

    def __get_img_resp(self):
        file_datas = []

        for reps in self.incoming_message_photo:
            file_id = str(reps['file_id'])
            img_info_res = requests.get(TELEGRAM_GET_PHOTO_INFO_URL.format(file_id))
            file_info = json.loads(img_info_res.content)

            file_url = TELEGRAM_GET_PHOTO_DATA_URL.format(file_info['result']['file_path'])
            file_data = requests.get(file_url)

            file_datas.append(file_data)

        return file_datas

    #download file from reponse
    def __save_images(self, file_datas):
        file_path = ''
        file_extention = ".png"
        is_success = False
        folder_path = ''
        file_index = 0
        ex = ''

        try:
            folder_path = create_folder(Constants.CRNN_OUPUT_DATA_DIRECTORY, self.chat_id)
            for file_data in file_datas:
                file_index = file_index + 1
                file_name = str(self.chat_id) + '_' + str(file_index)

                file_path = create_file_path(folder_path, file_name, file_extention)

                with open(file_path, 'wb') as f:
                    f.write(file_data.content)
                    print('file save in ', file_path)

            is_success = True
        except Exception as e:
            ex = str(e)

        return is_success, folder_path, ex, file_path

    @staticmethod
    def init_webhook(url):
        requests.get(url)


