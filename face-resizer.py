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
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


# --- Output ---
plt.imshow(img)
plt.show()
