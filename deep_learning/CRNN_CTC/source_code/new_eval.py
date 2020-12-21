import fnmatch
import os
import sys
import ntpath
import string
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

char_list = string.ascii_letters + string.digits

parser = argparse.ArgumentParser(description="Text Recognize")
parser.add_argument('--train_folder_path', dest='train_folder_path')
parser.add_argument('--valid_folder_path', dest='valid_folder_path')
parser.add_argument('--output_test_folder_path', dest='output_test_folder_path')
parser.add_argument('--max_train_files', dest='max_train_files')
parser.add_argument('--mode', dest='mode')
parser.add_argument('--model_file_path', dest='model_file_path')
_args = parser.parse_args()
train_folder_path_arg = _args.train_folder_path
valid_folder_path_arg = _args.valid_folder_path
output_test_folder_path_arg = _args.output_test_folder_path
max_train_files_arg = int(_args.max_train_files) or 15000
mode_arg = _args.mode or "train"
model_file_path_arg = _args.model_file_path


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def encode_to_labels(text):
    digit_list = []
    for index, char in enumerate(text):
        try:
            digit_list.append(char_list.index(char))
        except:
            print(char)

    return digit_list


class TextRecognize(object):
    def __init__(self, train_folder_path, valid_folder_path, output_test_folder_path, model_file_path,
                 max_train_files=10,
                 max_label_length=0, mode="train"):

        self.train_folder_path = train_folder_path
        self.valid_folder_path = valid_folder_path
        self.output_test_folder_path = output_test_folder_path
        self.max_label_length = max_label_length
        self.train_image_array = []
        self.train_text_array = []
        self.train_input_length = []
        self.train_label_length = []
        self.origin_text_array = []

        self.valid_image_array = []
        self.valid_text_array = []
        self.valid_input_length = []
        self.valid_label_length = []
        self.valid_origin_text_array = []
        self.max_train_files = max_train_files
        self.train_files_count = 0
        self.model = None
        self.mode = mode
        self.act_model = None
        self.train_padded_txt = None
        self.valid_padded_txt = None
        self.model_file_path = model_file_path
        self.init_data()
        self.init_model()
        self.run_function()

    def run_function(self):
        print("run_function")
        if "train" in self.mode:
            self.train()
        if "predict" in self.mode:
            self.predict()

    def pre_process_image(self, file_path):
        filename = ntpath.basename(file_path)
        image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
        # convert each image of shape (32, 128, 1)
        w, h = image.shape
        if h > 128 or w > 32:
            return
        if w < 32:
            add_zeros = np.ones((32 - w, h)) * 255
            image = np.concatenate((image, add_zeros))

        if h < 128:
            add_zeros = np.ones((32, 128 - h)) * 255
            image = np.concatenate((image, add_zeros), axis=1)
        image = np.expand_dims(image, axis=2)

        # Normalize each image
        image = image / 255.

        # get the text from the image
        text = filename.split('_')[1]

        # compute maximum length of the text
        if len(text) > self.max_label_length:
            self.max_label_length = len(text)
        # print(text, image)
        # print(text)
        return {
            "text": text,
            "image": image
        }

    def init_data(self):
        try:
            print("init_data")
            path = "{}/*.jpg"
            flag = 0
            if self.train_folder_path:
                for root, dirnames, filenames in os.walk(path):
                    for train_file in fnmatch.filter(filenames, '*.jpg'):
                        preprocessed_image = self.pre_process_image(train_file)
                        if preprocessed_image:
                            # print('train', preprocessed_image["text"])
                            self.origin_text_array.append(preprocessed_image["text"])
                            self.train_label_length.append(len(preprocessed_image["text"]))
                            self.train_input_length.append(31)
                            self.train_image_array.append(preprocessed_image["image"])
                            self.train_text_array.append(encode_to_labels(preprocessed_image["text"]))
                            if self.train_files_count == self.max_train_files:
                                flag = 1
                                break
                            self.train_files_count += 1
                    if flag == 1:
                        break

            #     path = path.format(self.train_folder_path)
            #     train_files = glob.glob(path)
            #     # print(type(train_files))
            #     for train_file in train_files:
            #
            #         preprocessed_image = self.pre_process_image(train_file)
            #         if preprocessed_image:
            #             # print('train', preprocessed_image["text"])
            #             self.origin_text_array.append(preprocessed_image["text"])
            #             self.train_label_length.append(len(preprocessed_image["text"]))
            #             self.train_input_length.append(31)
            #             self.train_image_array.append(preprocessed_image["image"])
            #             self.train_text_array.append(encode_to_labels(preprocessed_image["text"]))
            #             # if self.train_files_count == self.max_train_files:
            #             #     break
            #             # self.train_files_count += 1

            if self.valid_folder_path:
                path = path.format(self.valid_folder_path)
                valid_files = glob.glob(path)
                for valid_file in valid_files:
                    preprocessed_image = self.pre_process_image(valid_file)

                    if preprocessed_image:
                        self.valid_origin_text_array.append(preprocessed_image["text"])
                        self.valid_label_length.append(len(preprocessed_image["text"]))
                        self.valid_input_length.append(31)
                        self.valid_image_array.append(preprocessed_image["image"])
                        self.valid_text_array.append(encode_to_labels(preprocessed_image["text"] or ''))

        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def init_model(self):
        print("init_model")

        # input with shape of height=32 and width=128
        inputs = Input(shape=(32, 128, 1))

        # convolution layer with kernel size (3,3)
        conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        # pooling layer with kernel size (2,2)
        pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

        conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
        pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

        conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

        conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
        # pooling layer with kernel size (2,1)
        pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

        conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
        # Batch normalization layer
        batch_norm_5 = BatchNormalization()(conv_5)

        conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
        batch_norm_6 = BatchNormalization()(conv_6)
        pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

        conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

        squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
        blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
        blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

        outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)

        # model to be used at test time
        act_model = Model(inputs, outputs)

        act_model.summary()

        labels = Input(name='the_labels', shape=[self.max_label_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
            [outputs, labels, input_length, label_length])
        model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

        self.act_model = act_model
        self.model = model
        return {
            "act_model": act_model,
            "model": model
        }

    def train(self):
        print("train")
        filepath = self.model_file_path
        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        callbacks_list = [checkpoint]
        # print(self.train_input_length)
        # print(self.train_label_length)
        # print(self.valid_input_length)
        # print(self.valid_label_length)
        self.train_padded_txt = pad_sequences(self.train_text_array,
                                              maxlen=self.max_label_length,
                                              padding='post',
                                              value=len(char_list))
        self.valid_padded_txt = pad_sequences(self.valid_text_array,
                                              maxlen=self.max_label_length,
                                              padding='post',
                                              value=len(char_list))
        training_img = np.array(self.train_image_array)

        train_input_length = np.array(self.train_input_length)

        train_label_length = np.array(self.train_label_length)

        valid_img = np.array(self.valid_image_array)
        valid_input_length = np.array(self.valid_input_length)
        valid_label_length = np.array(self.valid_label_length)
        batch_size = 256
        epochs = 30
        # print(training_img)

        self.model.fit(x=[training_img,
                          self.train_padded_txt,
                          train_input_length,
                          train_label_length],
                       y=np.zeros(len(training_img)),
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=([valid_img,
                                         self.valid_padded_txt,
                                         valid_input_length,
                                         valid_label_length],
                                        [np.zeros(len(valid_img))]),
                       verbose=1, callbacks=callbacks_list)

        return

    def predict(self):
        self.act_model.load_weights(self.model_file_path)

        # predict outputs on validation images
        valid_img = np.array(self.valid_image_array)

        prediction = self.act_model.predict(valid_img[:10])

        # use CTC decoder
        out = K.get_value(K.ctc_decode(prediction,
                                       input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                                       greedy=True)[0][0])

        # see the results
        index = 0
        predict_results = []
        for x in out:
            predict_text = ""
            for p in x:
                if int(p) != -1:
                    predict_text += char_list[int(p)]
            predict_results.append({
                "origin_text": self.valid_origin_text_array[index],
                "predict_text": predict_text
            })
            # print("original_text =  ", self.valid_origin_text_array[index])
            # print("predicted text = ", end='')

            index += 1
        print(predict_results)
        return predict_results


tr = TextRecognize(train_folder_path=train_folder_path_arg or None,
                   valid_folder_path=valid_folder_path_arg,
                   max_train_files=max_train_files_arg,
                   mode=mode_arg,
                   output_test_folder_path=output_test_folder_path_arg,
                   model_file_path=model_file_path_arg)
