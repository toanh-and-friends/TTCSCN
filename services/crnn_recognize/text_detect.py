import sys
import ntpath
import string
import glob
import cv2
import numpy as np
import keras.backend as K
import time

from services.crnn_recognize.model_singleton.crnn_model_singleton import CrnnSingleton

char_list = string.ascii_letters + string.digits

batch_size_arg =  256
epochs_arg =  10
max_train_files_arg =  15000
mode_arg =  "train"
trained_flag = 0

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
        self.model, self.act_model = CrnnSingleton.getModel()

    def pre_process_image(self, file_path):
        try: 
            now = int(round(time.time() * 1000))
            filename = ntpath.basename(file_path)
            filename_split = filename.split('_')
            text = "0"
            if (len(filename_split) > 1): 
                text = filename_split[1]
            image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
            # cv2.imwrite("../datasets/output_test_data/{}-{}-gray.jpg".format(now,text), image)
            # convert each image of shape (32, 128, 1)
            # print(image)
            w, h = image.shape
            
            # print(w,h)
            if h > 128 or w > 32:
                image = cv2.resize(image, (128,32), interpolation = cv2.INTER_AREA)
                w, h = image.shape
                # return
            if w < 32:
                add_zeros = np.ones((32 - w, h)) * 255
                image = np.concatenate((image, add_zeros))
 
            if h < 128:
                add_zeros = np.ones((32, 128 - h)) * 255
                image = np.concatenate((image, add_zeros), axis=1)
            image = np.expand_dims(image, axis=2)
 
            # Normalize each image
            
            
            cv2.imwrite("../datasets/output_test_data/{}-{}.jpg".format(now,text), image)
            # get the text from the image
 
            image = image / 255.
 
            # compute maximum length of the text
            if len(text) > self.max_label_length:
                self.max_label_length = len(text)
            # print(text, image)
            return {
                "text": text,
                "image": image
            }
        except Exception as e:
            print({"error":e})

    def init_data(self):
        try:
            print("init_data")
            path = "{}/*.jpg"
            # path = path.format(self.train_folder_path)
            flag = 0

            if self.valid_folder_path:
                path = path.format(self.valid_folder_path)
                valid_files = glob.glob(path)
                for valid_file in valid_files:
                    print('preprocessed_image start ',valid_file) 
                    preprocessed_image = self.pre_process_image(valid_file)
                    print('preprocessed_image end ',preprocessed_image)
                    if preprocessed_image:
                        print(valid_file)
                        self.valid_origin_text_array.append(preprocessed_image["text"])
                        self.valid_label_length.append(len(preprocessed_image["text"]))
                        self.valid_input_length.append(31)
                        self.valid_image_array.append(preprocessed_image["image"])
                        self.valid_text_array.append(encode_to_labels(preprocessed_image["text"] or ''))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

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
        predict_results = ""
        for x in out:
            predict_text = ""
            for p in x:
                if int(p) != -1:
                    predict_text += char_list[int(p)]

            predict_results = predict_results + " " + predict_text
            # print("original_text =  ", self.valid_origin_text_array[index])
            # print("predicted text = ", end='')

            index += 1
        print("crnn predict output: ", predict_results)
        return predict_results
