import cv2
import pytesseract
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

BASE_DIR = os.path.dirname(__file__)

root = tk.Tk()
root.withdraw()

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

def detect_username(path=None, messages=False):
    if path == None:
        path = filedialog.askopenfilename(initialdir="images")
        path = os.path.join(BASE_DIR, path)

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract"
    image = cv2.imread(path)
    cropped = image[200:250, 120:600]
    text = pytesseract.image_to_string(cropped, lang="spa")
    username = text.split()[0]

    if messages:
        print("Username: ", username)

    gray = get_grayscale(cropped)
    thresh = thresholding(gray)
    opening(gray)
    canny(gray)

    h, w, c = cropped.shape
    boxes = pytesseract.image_to_boxes(cropped)
    for b in boxes.splitlines():
        b = b.split(" ")
        cropped = cv2.rectangle(cropped, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    # cv2.imshow("Cropped image", cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return username


if "__main__" == __name__:
    print(detect_username())