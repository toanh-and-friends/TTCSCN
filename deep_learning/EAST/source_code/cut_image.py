from datetime import date
import numpy as np
import cv2
import os

# load the image and show it
# image = cv2.imread("../datasets/ICDAR2015/test_data_output/text.png")
# print(image.shape)
# cropped = image[124:162,124:236]
# # cv2.imshow("cropped", cropped)
# cv2.imwrite("cut_image.png", cropped)
# print('Done')
# cv2.waitKey(0)

def find_max_min_x_y(array):
    max_x = array[0]
    max_y = array[0]
    min_x = array[0]
    min_y = array[0]
    for i in range(0, len(array)):
        if i % 2 == 0:
            if max_x < array[i]:
                max_x = array[i]
            if min_x > array[i]:
                min_x = array[i]
        else:
            if max_y < array[i]:
                max_y = array[i]
            if min_y > array[i]:
                min_y = array[i]
    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y
    }


class ImagesData(object):
    def __init__(self, output_folder_path, image_file_path, data_image_file_path):
        self.output_folder_path = output_folder_path
        self.image_file_path = image_file_path
        self.data_image_file_path = data_image_file_path
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)
    def read_data_image_file(self):
        try:
            data = []
            f = open(self.data_image_file_path)
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '')
                line = line.split(',')
                data.append([int(numeric_string) for numeric_string in line])

            # data = data[np.argsort(data[:, 0])]
            data = np.array(data)
            # data = np.argsort(data[:, 1])
            # a[a[:, -1].argsort()]
            ind = np.lexsort((data[:, 1], data[:, 0]))

            print(data[ind])
            # print(data)
            return data
        except:
            print('read_data_image_file error')
            return []

    def crop_image(self):
        image_index = 0
        image = cv2.imread(self.image_file_path)
        image_data = self.read_data_image_file()
        for image_data_item in image_data:
            max_min_x_y = find_max_min_x_y(image_data_item)
            max_x = max_min_x_y["max_x"]
            min_x = max_min_x_y["min_x"]
            max_y = max_min_x_y["max_y"]
            min_y = max_min_x_y["min_y"]
            print([min_y,max_y, min_x,max_x])
            cropped = image[min_y:max_y, min_x:max_x]
            # cv2.imshow("cropped", cropped)
            cv2.imwrite("{}/{}.png".format(self.output_folder_path, image_index), cropped)
            image_index = image_index + 1


ig = ImagesData(output_folder_path='../datasets/ICDAR2015/test_data_output/1',
                image_file_path='../datasets/ICDAR2015/test_data/images.jpeg',
                data_image_file_path='../datasets/ICDAR2015/test_data_output/images.txt')
ig.crop_image()
