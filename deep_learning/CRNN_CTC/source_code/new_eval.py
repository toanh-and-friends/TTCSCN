import fnmatch
import os
import ntpath
import string
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

char_list = string.ascii_letters + string.digits


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


class TextRecognize(object):
    def __init__(self, train_folder_path, valid_folder_path, output_test_folder_path, max_train_files=0,
                 max_label_length=0, mode="train"):
        self.train_folder_path = train_folder_path
        self.valid_folder_path = valid_folder_path
        self.output_test_folder_path =output_test_folder_path
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
        self.valid_origin_text = []
        self.max_train_files = max_train_files
        self.train_files_count = 0
        self.model = None
        self.mode = mode
        self.act_model = None
        self.train_padded_txt = None
        self.valid_padded_txt = None
        self.init_data(self.mode)
        self.init_model()

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

        return {
            "text": text,
            "image": image
        }

    def init_data(self):
        try:
            path = "{}/*.jpg"

            if "train" in self.mode:
                path = path.format(self.train_folder_path)
                train_files = glob.glob(path)
                for train_file in range(train_files):
                    preprocessed_image = self.pre_process_image(train_file)
                    self.origin_text_array.append(preprocessed_image["text"])
                    self.train_label_length.append(len(preprocessed_image["text"]))
                    self.train_input_length.append(31)
                    self.train_image_array.append(preprocessed_image["image"])
                    self.train_text_array.append(self.encode_to_labels(preprocessed_image["text"]))
                    if self.train_files_count == self.max_train_files:
                        break
                    self.train_files_count += 1
            if "valid" in self.mode:
                path = path.format(self.valid_folder_path)
                valid_files = glob.glob(path)
                for valid_file in range(valid_files):
                    preprocessed_image = self.pre_process_image(valid_file)
                    self.valid_origin_text_array.append(preprocessed_image["text"])
                    self.valid_train_label_length.append(len(preprocessed_image["text"]))
                    self.vaild_train_input_length.append(31)
                    self.valid_train_image_array.append(preprocessed_image["image"])
                    self.vaild_train_text_array.append(self.encode_to_labels(preprocessed_image["text"]))
        except:
            return

    @staticmethod
    def encode_to_labels(self, text):
        digit_list = []
        for index, char in enumerate(text):
            try:
                digit_list.append(char_list.index(char))
            except:
                print(char)
            finally:
                print("finally")

        return digit_list

    def init_model(self):
        self.train_padded_txt = pad_sequences(self.train_text_array,
                                              maxlen=self.max_label_length,
                                              padding='post',
                                              value=len(char_list))
        self.valid_padded_txt = pad_sequences(self.vaild_train_text_array,
                                              maxlen=self.max_label_length,
                                              padding='post',
                                              value=len(char_list))
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
        filepath = "../models/best_model.hdf5"
        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        callbacks_list = [checkpoint]

        training_img = np.array(self.train_image_array)
        train_input_length = np.array(self.train_input_length)
        train_label_length = np.array(self.train_label_length)

        valid_img = np.array(self.valid_image_array)
        valid_input_length = np.array(self.valid_input_length)
        valid_label_length = np.array(self.valid_label_length)

        batch_size = 256
        epochs = 30
        self.model.fit(x=[training_img, self.train_padded_txt, train_input_length, train_label_length],
                       y=np.zeros(len(training_img)), batch_size=batch_size, epochs=epochs, validation_data=(
            [valid_img, self.valid_padded_txt, valid_input_length, valid_label_length], [np.zeros(len(valid_img))]),
                       verbose=1, callbacks=callbacks_list)

        return

    def predict(self):
        self.act_model.load_weights('../models/best_model.hdf5')

        # predict outputs on validation images
        prediction = self.act_model.predict(self.valid_image_array)

        # use CTC decoder
        out = K.get_value(K.ctc_decode(prediction,
                                       input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                                       greedy=True)[0][0])

        # see the results
        index = 0
        for x in out:
            print("original_text =  ", self.valid_origin_text_array[index])
            print("predicted text = ", end='')
            for p in x:
                if int(p) != -1:
                    print(char_list[int(p)], end='')
            print('\n')
            index += 1
        return
