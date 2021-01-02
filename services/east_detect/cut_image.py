from datetime import date
import numpy as np
import cv2
import os
import shapely
from shapely.geometry import LineString, Point
from imutils import contours
from skimage import measure


def point_of_intersection(point_a, point_b, point_c, point_d):
    line1 = LineString([Point(point_a), Point(point_b)])
    # print(line1)
    line2 = LineString([Point(point_c), Point(point_d)])
    # print(line2)
    int_pt = line1.intersection(line2)
    # print(int_pt)
    return {
        "x": int_pt.x,
        "y": int_pt.y
    }


def find_max_min_x_y(array):
    max_x = array[0]
    max_y = array[1]
    min_x = array[0]
    min_y = array[1]
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


def blur(image_file_path, kernel_width, kernel_height):
    if not os.path.isfile(image_file_path):
        raise Exception('File not found @ %s' % image_file_path)

    img = cv2.imread(image_file_path)

    blur_img = cv2.blur(img, ksize=(kernel_width, kernel_height))  # or cv2.boxFilter

    new_image_file_path = 'blur_%s_%d_%d.jpeg' % (
        os.path.splitext(os.path.basename(image_file_path))[0], kernel_width, kernel_height)
    cv2.imwrite(new_image_file_path, blur_img)

    return {
        "image_file_path": new_image_file_path
    }


def gray(image_file_path):
    if not os.path.isfile(image_file_path):
        raise Exception('File not found @ %s' % image_file_path)
    img = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2GRAY)
    new_image_file_path = 'gray_%s.jpg' % (
        os.path.splitext(os.path.basename(image_file_path))[0])
    cv2.imwrite(new_image_file_path, img)
    return {
        "image_file_path": new_image_file_path
    }


class ImagesData(object):
    def __init__(self, output_folder_path, image_file_path, data_image_file_path):
        self.output_folder_path = output_folder_path
        self.image_file_path = image_file_path
        self.data_image_file_path = data_image_file_path
        if not os.path.isfile(image_file_path):
            raise Exception('File not found @ %s' % image_file_path)
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)

    def detect_baselines_crop_images(self):
        img = cv2.cvtColor(cv2.imread(self.image_file_path), cv2.COLOR_BGR2GRAY)

        # convert each image of shape (32, 128, 1)
        height, width = img.shape
        # print(width,height)
        blank_image = np.zeros((height, width, 3), np.uint8)
        blank_image[:] = (0, 0, 0)
        # cv2.imwrite("test.png", blank_image)
        image_data = self.read_data_image_file()
        # print(image_data)
        for space in image_data:
            delta_space = space.copy()
            delta_space_height = (delta_space[7] - delta_space[1]) * 0.30
            delta_space[1] = delta_space[1] + delta_space_height
            delta_space[3] = delta_space[3] + delta_space_height
            delta_space[5] = delta_space[5] - delta_space_height
            delta_space[7] = delta_space[7] - delta_space_height
            pts = np.array([[delta_space[0], delta_space[1]], [delta_space[2], delta_space[3]], [delta_space[4], delta_space[5]], [delta_space[6], delta_space[7]]],
                           np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(blank_image, [pts], (255, 255, 255))
        # kernel = np.ones((4, 4), np.uint8)
        blurred = cv2.GaussianBlur(blank_image, (11, 11), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(25, 1)) #x
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1, 2)) #y
        thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, kernel2, iterations=2)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        cv2.imwrite("step0.jpg", thresh)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # erode_image = cv2.erode(blank_image, kernel2, iterations=2)
        # dilate_image = cv2.dilate(erode_image, kernel, iterations=1)

        # cv2.imwrite("test.png", thresh)
        # kernel_width = 5
        # kernel_height = 5
        # kernel = np.ones((10, 10), np.uint8)
        # img = cv2.cvtColor(cv2.imread(self.image_file_path), cv2.COLOR_BGR2GRAY)
        # blur_img = cv2.blur(img, ksize=(kernel_width, kernel_height))
        # dilation = cv2.dilate(blur_img, kernel, iterations=1)
        # _gray = cv2.bitwise_not(dilation)
        # bw = cv2.adaptiveThreshold(_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        # Show binary image
        # show_wait_destroy("binary", bw)
        # cv2.imwrite("test.png", bw  )
        # cv2.imwrite("test.png", dilation)
        # print('Done')

        lem = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        lem = cv2.dilate(lem, kernel)
        contour, hier = cv2.findContours(lem.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # And fill them
        for c, h in zip(contour, hier[0]):
            if h[3] != -1:
                cv2.drawContours(lem, [c], 0, 255, -1)

        # Now bring the leming back to its original size
        lem = cv2.erode(lem, kernel)

        # Remove the cord by wiping-out all vertical lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)) #15

        lem = cv2.morphologyEx(lem, cv2.MORPH_OPEN, kernel)

        # Find the contour of the leming
        contour, _ = cv2.findContours(lem.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # And draw it on the original image
        lines = []
        line_index = 0
        # cv2.imwrite("step1.png", thresh)
        frames = []

        for c in contour:

            lines.append([])
            # enter your filtering here
            x, y, w, h = cv2.boundingRect(c)
            frames.append([x, y, w, h])

        frames = sorted(frames , key=lambda k: [k[1], k[0]])
        for frame in frames:
            x, y, w, h = frame[0], frame[1], frame[2], frame[3]
            print(x, y, w, h)
            for space in image_data:
                # print(space)
                # cv2.line(thresh, (space[0], space[1]), (space[4], space[5]), (0, 0, 255), thickness=1)
                # cv2.line(thresh, (space[2], space[3]), (space[6], space[7]), (0, 0, 255), thickness=1)
                # thresh[space[1], space[0]] = [0, 255, 0]
                # thresh[space[7], space[6]] = [0, 255, 0]
                middle_point = point_of_intersection(point_a=(space[0], space[1]),
                                                     point_b=(space[4], space[5]),
                                                     point_c=(space[2], space[3]),
                                                     point_d=(space[6], space[7]))
                middle_point_x = middle_point["x"]
                middle_point_y = middle_point["y"]
                # print(middle_point_x, middle_point_y)
                # thresh[int(middle_point_y), int(middle_point_x)] = [0, 255, 0]
                if x <= middle_point_x <= x + w and y <= middle_point_y <= y + h:
                    lines[line_index].append(space)
            cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 0), 1)
            line_index = line_index + 1

        cv2.imwrite("step2.jpg", thresh)

        for line_index in range(0, len(lines)):
            for item_index in range(0, len(lines[line_index])):
                for _item_index in range(0, len(lines[line_index]) - item_index - 1):
                    # print(lines[_item_index][0], lines[_item_index+1][0])
                    if lines[line_index][_item_index][0] > lines[line_index][_item_index + 1][0]:
                        lines[line_index][_item_index], lines[line_index][_item_index + 1] = lines[line_index][_item_index +1], lines[line_index][_item_index]
        # print(lines)
        image = cv2.imread(self.image_file_path)
        image_index = 0
        for line_index in range(0, len(lines)):
            for item_index in range(0, len(lines[line_index])):
                max_min_x_y = find_max_min_x_y(lines[line_index][item_index])
                max_x = max_min_x_y["max_x"]
                min_x = max_min_x_y["min_x"]
                max_y = max_min_x_y["max_y"]
                min_y = max_min_x_y["min_y"]
                print("min_y, max_y, min_x, max_x ",[min_y, max_y, min_x, max_x])
                cropped = image[min_y:max_y, min_x:max_x]
                # cv2.imshow("cropped", cropped)
                cv2.imwrite("{}/{}.jpg".format(self.output_folder_path, image_index), cropped)

                image_index = image_index + 1


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

            # print(data[ind])
            # print(data)
            return data
        except:
            print('read_data_image_file error')
            return []

    # def crop_image(self):
    #     image_index = 0
    #     image = cv2.imread(self.image_file_path)
    #     image_data = self.read_data_image_file()
    #     # pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    #     # pts = pts.reshape((-1, 1, 2))
    #     # cv.polylines(img, [pts], True, (0, 255, 255))
    #     for image_data_item in image_data:
    #         max_min_x_y = find_max_min_x_y(image_data_item)
    #         max_x = max_min_x_y["max_x"]
    #         min_x = max_min_x_y["min_x"]
    #         max_y = max_min_x_y["max_y"]
    #         min_y = max_min_x_y["min_y"]
    #         # print([min_y, max_y, min_x, max_x])
    #         cropped = image[min_y:max_y, min_x:max_x]
    #         # cv2.imshow("cropped", cropped)
    #         # cv2.imwrite("{}/{}.png".format(self.output_folder_path, image_index), cropped)

    #         image_index = image_index + 1

