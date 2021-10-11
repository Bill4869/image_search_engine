import cv2
# import pytesseract
from easyocr import Reader

def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image to be OCR'd")
# ap.add_argument("-l", "--langs", type=str, default="en",
# 	help="comma separated list of languages to OCR")
# ap.add_argument("-g", "--gpu", type=int, default=-1,
# 	help="whether or not GPU should be used")
# args = vars(ap.parse_args())


image = cv2.imread('./1.jpg', cv2.IMREAD_COLOR)
image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)

# tesseract
# By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
# we need to convert from BGR to RGB format/mode:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\BiLL\AppData\Local\Tesseract-OCR\tesseract.exe'
# img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
# print(pytesseract.image_to_string(img_rgb))

# easyOCR
reader = Reader(['en', 'ja'], gpu = False)
results = reader.readtext(image)

for (bbox, text, prob) in results:
	# display the OCR'd text and associated probability
	print("[INFO] {:.4f}: {}".format(prob, text))
	# unpack the bounding box
	(tl, tr, br, bl) = bbox
	tl = (int(tl[0]), int(tl[1]))
	tr = (int(tr[0]), int(tr[1]))
	br = (int(br[0]), int(br[1]))
	bl = (int(bl[0]), int(bl[1]))
	# cleanup the text and draw the box surrounding the text along
	# with the OCR'd text itself
	text = cleanup_text(text)
	cv2.rectangle(image, tl, br, (0, 255, 0), 2)
	cv2.putText(image, text, (tl[0], tl[1] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
