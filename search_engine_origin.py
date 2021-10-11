import glob, os, time
import cv2
import numpy as np
import lpips, fid_mod
import torchvision.transforms.functional as T
from PIL import Image 
from easyocr import Reader
from tqdm import tqdm
import tensorflow_hub as hub
import pandas as pd
import tensorflow as tf


def get_img(path):
    # filenames = glob.glob(path + "*.jpg")
    file_paths = []
    for folder, sub_folders, files in os.walk(path):
        for special_file in files:
            if '.jpg' in special_file:
                file_paths.append(os.path.join(folder, special_file))
    
    return file_paths

def get_color_tag(img, color):
    colors = {'青': np.uint8([[[255, 0, 0]]]), '緑': np.uint8([[[0, 255, 0]]]),
            '赤': np.uint8([[[0, 0, 255]]])}

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    tag = {}
        
    color_hsv = cv2.cvtColor(colors[color], cv2.COLOR_BGR2HSV)
    lower = np.array([color_hsv[0, 0, 0] - 10, 100, 100])
    upper = np.array([color_hsv[0, 0, 0] + 10, 255, 255])

    mask = cv2.inRange(img_hsv, lower, upper)

    retval = cv2.countNonZero(mask)
    percent = retval * 100 / (img.shape[0] * img.shape[1])

    if(percent > 0):
        tag[color] = percent
    
    return tag

def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text])

def get_shape_tag(gray, shape):
    ret, threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    
    tag = {}

    # list for storing names of shapes
    for contour in contours:
        # here we are ignoring first counter because 
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue
    
        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        
        if len(approx) == 3 and shape == '三角形':
            if('三角形' not in tag):
                tag['三角形'] = 1
            else:
                tag['三角形'] += 1
    
        elif len(approx) == 4 and shape == '四角形':
            if('四角形' not in tag):
                tag['四角形'] = 1
            else:
                tag['四角形'] += 1
        elif len(approx) == 5 and shape == '五角形':
            if('五角形' not in tag):
                tag['五角形'] = 1
            else:
                tag['五角形'] += 1
        elif len(approx) == 6 and shape == '六角形':
            if('六角形' not in tag):
                tag['六角形'] = 1
            else:
                tag['六角形'] += 1
        else:
            if('丸' not in tag):
                tag['丸'] = 1
            else:
                tag['丸'] += 1
    return tag
def get_text_tag(reader, img, txt=None):
    results = reader.readtext(img)

    tag = {}
    for (bbox, text, prob) in results:
        if text == txt:
            tag[text] = prob
    return tag

def cal_fid(source, target):
    img = [[source], [target]]
    fid_value = fid_mod.calculate_fid_given_paths(img)
    tag = {'fid': fid_value}
    return tag

def cal_lpips(loss_fn_vgg, source, target):
    img0 = Image.open(source)
    img0 = img0.resize((128, 128))
    img0 = (T.to_tensor(img0) - 0.5) * 2
    img0.unsqueeze(0)

    img1 = Image.open(target)
    img1 = img1.resize((128, 128))
    img1 = (T.to_tensor(img1) - 0.5) * 2
    img1.unsqueeze(0)

    d = loss_fn_vgg(img0, img1)
    val = d.detach().numpy()
    tag = {'lpips': val[0, 0, 0, 0]}
    return tag

def get_obj_tag(img, detector, labels, obj):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rgb_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.uint8)
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)

    boxes, scores, classes, num_detections = detector(rgb_tensor)

    pred_labels = classes.numpy().astype('int')[0] 
    pred_labels = [labels[i] for i in pred_labels]
    pred_scores = scores.numpy()[0]

    tag = {}
    for score, label in zip(pred_scores, pred_labels):
        if score > 0.5 and label == obj:
            tag[label] = score
    return tag

def display(path, tag):
    blank_image = np.zeros((128,128,3), np.uint8)

    l = len(tag) // 4 + (len(tag) % 4 > 0)

    im_list = []
    for i in range(l):
        im = []
        for j in range(4):
            try:
                img = cv2.imread(path[tag[4*i+j]], cv2.IMREAD_COLOR)
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                im.append(img)
            except:
                im.append(blank_image)

        im_list.append(im)
    
    im_title = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list])

    cv2.imwrite('result.jpg', im_title)

def search(path, color, shape, obj, text, image, sort):
    file_paths = get_img(path)
    start = time.time()

    reader = Reader(['en', 'ja'], gpu = True)

    loss_fn_vgg = lpips.LPIPS(net='vgg', lpips=True)

    detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite4/detection/1")
    labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
    labels = labels['OBJECT (2017 REL.)']

    img_db = []
    tag_db = {}

    for index, file_path in enumerate(tqdm(file_paths)):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.blur(img_gray,(9,9))

        img_data = {}

        if color:
            color_tag = get_color_tag(img, color)
            if color_tag:
                img_data.update(color_tag)

        if text:
            word_tag = get_text_tag(reader, img, text)
            if word_tag:
                img_data.update(word_tag)

        if shape:
            shape_tag = get_shape_tag(img_blur, shape)
            if shape_tag:
                img_data.update(shape_tag)

        if obj:
            obj_tag = get_obj_tag(img, detector, labels, obj)
            if obj_tag:
                img_data.update(obj_tag)

        if(image is not None):
            fid_tag = cal_fid(image, file_path)
            img_data.update(fid_tag)

            lpips_tag = cal_lpips(loss_fn_vgg, image, file_path)
            img_data.update(lpips_tag)

        img_db.append(img_data)

        for tag in img_data:
            if(tag in tag_db):
                tag_db[tag].append(index)
            else:
                tag_db[tag] = [index]


    ans = None
    for tag in tag_db:
        if ans is None:
            ans = set(tag_db[tag])
        else:
            ans = ans & set(tag_db[tag])
    
    if('fid' in tag_db):
        ans = {i:(img_db[i]['fid'] + img_db[i]['lpips']) for i in ans}
        ans = {k: v for k, v in sorted(ans.items(), key=lambda item: item[1])}
        display(file_paths, list(ans.keys())[:8])
    elif sort:
        for tag in tag_db:
            ans = {i:img_db[i][tag] for i in ans}
            ans = {k: v for k, v in sorted(ans.items(), key=lambda item: item[1], reverse=True)}
            display(file_paths, list(ans.keys()))
    else:
        display(file_paths, list(ans))

    print('time: {}'.format(time.time() - start))
    for i in img_db:
        print(i)
    # for i in tag_db:
    #     print('{}: {}'.format(i, tag_db[i]))
    

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path", type=str, default='../images/mix', help='Path to images')
    parser.add_argument("-c", "--color", type=str, default=None, help='searching color')
    parser.add_argument("-s", "--shape", type=str, default=None, help='searching shape')
    parser.add_argument("-o", "--object", type=str, default=None, help='searching object')
    parser.add_argument("-t", "--text", type=str, default=None, help='searching text')
    parser.add_argument("-i", "--image", type=str, default=None, help='searching image')
    parser.add_argument("--sort", type=bool, default=False, help='sorting')

    args = parser.parse_args()

    search(args.path, args.color, args.shape, args.object, args.text, args.image, args.sort)
