import numpy as np
import cv2 as cv
import glob

img_path = "./logo/"
filenames = glob.glob(img_path + "*.jpg")

for index, path in enumerate(filenames):
    img = cv.imread(path)
    img = cv.resize(img, (256, 256), interpolation=cv.INTER_AREA)
    # img = cv.GaussianBlur(img, (3, 3), 0)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # sift = cv.SIFT_create()
    # kp = sift.detect(gray,None)

    # surf = cv.xfeatures2d.SURF_create(2000)
    # kp, des = surf.detectAndCompute(gray,None)

    # img=cv.drawKeypoints(gray,kp,img)
    # cv.imwrite('sift_keypoints.jpg',img)

    # star = cv.xfeatures2d.StarDetector_create()
    # brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    # kp = star.detect(gray,None)
    # kp, des = brief.compute(gray, kp)

    # orb = cv.ORB_create()
    # kp = orb.detect(gray,None)
    # kp, des = orb.compute(gray, kp)

    img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite('{}.jpg'.format(index), img)