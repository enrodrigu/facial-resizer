import cv2
import sys
from matplotlib import pyplot as plt

# Get model for detecting faces
face_cascade = cv2.CascadeClassifier('model\haarcascade_frontalface_default.xml')

# Read image
img = cv2.imread("data\\"+sys.argv[1])

# --- Detection ---
## Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
## Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

 ## Draw rectangle around faces
'''for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)'''

# --- Cropping ---
## Get size of image
height, width, channels = img.shape
## Cropping image
if (len(faces) != 0):
    (f_x, f_y, f_w, f_h) = faces[0]
else:
    f_x = 0
    f_y = 0
    f_w = width
    f_h = height
    print("No face detected")

### Cropping width
if (f_w*2 > width):
    margin_w = (width - f_w)//2
else:
    margin_w = f_w//2

### Cropping height
if (f_h*2 > height):
    margin_h = (height - f_h)//2
else:
    margin_h = f_h//2

### Cropping
img = img[f_y-margin_h:f_y+f_h+margin_h, f_x-margin_w:f_x+f_w+margin_w]
img = cv2.resize(img, (768, 768))

# --- Output ---
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.imwrite("out\\"+sys.argv[1], img)
