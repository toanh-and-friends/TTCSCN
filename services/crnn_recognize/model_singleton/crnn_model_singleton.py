import copy
import string
# from ..text_detect import ctc_lambda_func
import threading

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
import keras
import keras.backend as K

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

class CrnnSingleton:
   __model = None
   __act_model = None

   @staticmethod
   def getModel():
        if CrnnSingleton.__model is None or CrnnSingleton.__act_model is None:
            with threading.Lock():
                if CrnnSingleton.__model is None or CrnnSingleton.__act_model is None:
                    CrnnSingleton.__model, CrnnSingleton.__act_model = CrnnSingleton.__init_model()

        model =  keras.models.clone_model(CrnnSingleton.__model)
        act_model =  keras.models.clone_model(CrnnSingleton.__act_model)

        return model, act_model

   @staticmethod
   def __init_model():
       char_list = string.ascii_letters + string.digits

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

       labels = Input(name='the_labels', shape=[2000], dtype='float32')
       input_length = Input(name='input_length', shape=[1], dtype='int64')
       label_length = Input(name='label_length', shape=[1], dtype='int64')
       loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
           [outputs, labels, input_length, label_length])
       model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

       model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

       return model, act_model

