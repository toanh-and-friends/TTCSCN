import cv2

lemBGR = cv2.imread("test.png")
lem = cv2.cvtColor(lemBGR, cv2.COLOR_BGR2GRAY)

# Dilate the image in order to close any external contour of the leming
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
lem = cv2.dilate(lem,kernel)

# Identify holes in the leming contour
# This could be done by iterative morphological operations,
# but this is not directly implemented in OpenCV
contour,hier = cv2.findContours(lem.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
# And fill them
for c,h in zip(contour, hier[0]):
    if h[3]!=-1:
        cv2.drawContours(lem,[c],0,255,-1)

# Now bring the leming back to its original size
lem = cv2.erode(lem,kernel)

# Remove the cord by wiping-out all vertical lines
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,1))
#lem = cv2.erode(lem,kernel) # first wipe-out
#lem = cv2.dilate(lem,kernel) # then bring back to original size
# erode and then dilate is the same as opening
lem = cv2.morphologyEx(lem,cv2.MORPH_OPEN,kernel)

# Find the contour of the leming
contour,_ = cv2.findContours(lem.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# And draw it on the original image
for c in contour:
    # enter your filtering here
    x,y,w,h = cv2.boundingRect(c)
    print(x,y,w,h)
    cv2.rectangle(lemBGR,(x,y),(x+w,y+h),(0,255,0),2)

# Display the result
# cv2.imshow("lem",lemBGR)
# cv2.waitKey()

cv2.imwrite("lem-res.png",lemBGR)