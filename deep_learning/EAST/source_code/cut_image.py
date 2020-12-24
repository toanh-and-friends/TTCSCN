import cv2
# load the image and show it
image = cv2.imread("../datasets/ICDAR2015/test_data_output/text.png")
print(image.shape)
cropped = image[10:162,1:145]
# cv2.imshow("cropped", cropped)
cv2.imwrite("cut_image.png", cropped)
print('Done')
# cv2.waitKey(0)