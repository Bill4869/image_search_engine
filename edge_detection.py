import cv2
import numpy as np
from matplotlib import pyplot as plt

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged
  
img = cv2.imread('../images/bld/bld1.jpg')
# img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blur
# gray = cv2.blur(gray,(5,5))
# gray = cv2.GaussianBlur(gray, (5, 5), 0)
# gray = cv2.medianBlur(gray,5)
# gray = cv2.bilateralFilter(gray, 9, 75, 75)

# edges = cv2.Canny(gray,30,150)
edges = auto_canny(gray, 0.3)

cv2.imshow('', img)
plt.imshow(edges, cmap = 'gray')
plt.show()
