import cv2
import glob
import numpy as np

img_path = "../images/cloth/"
filenames = glob.glob(img_path + "*.jpg")

colors = {'青': np.uint8([[[255, 0, 0]]]), '緑': np.uint8([[[0, 255, 0]]]),
            '赤': np.uint8([[[0, 0, 255]]])}

img_db = []
tag_db = {}

for index, path in enumerate(filenames):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    tag = {}
    
    for color in colors:
        color_hsv = cv2.cvtColor(colors[color], cv2.COLOR_BGR2HSV)
        lower = np.array([color_hsv[0, 0, 0] - 10, 100, 100])
        upper = np.array([color_hsv[0, 0, 0] + 10, 255, 255])

        mask = cv2.inRange(img_hsv, lower, upper)

        retval = cv2.countNonZero(mask)
        percent = retval * 100 / (img.shape[0] * img.shape[1])

        if(percent > 0):
            tag[color] = percent

    
        raise




