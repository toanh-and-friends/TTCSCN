import os
import cv2

INPUT = '../datasets/ICDAR2015/test_data/stop.jpeg'
KERNEL_WIDTH = 10
KERNEL_HEIGHT = 10

if not os.path.isfile(INPUT):
    raise Exception('File not found @ %s' % INPUT)

img = cv2.imread(INPUT)


blur_img = cv2.blur(img, ksize=(KERNEL_WIDTH, KERNEL_HEIGHT)) # or cv2.boxFilter

cv2.imwrite('box_blur_%s_%d_%d.jpeg' % (os.path.splitext(os.path.basename(INPUT))[0], KERNEL_WIDTH, KERNEL_HEIGHT), blur_img)