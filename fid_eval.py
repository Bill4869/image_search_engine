import fid_mod

import cv2
import glob
import numpy as np

img_path = "./dog/"
filenames = glob.glob(img_path + "*.jpg")
filenames.sort()

for i in range(len(filenames)):
    for j in range(i + 1, len(filenames)):
        img = [[filenames[i]], [filenames[j]]]
        fid_value = fid_mod.calculate_fid_given_paths(img)
        print(filenames[i], filenames[j])
        print("FID: ", fid_value)
