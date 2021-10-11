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
from feature_extractor import FeatureExtractor
from tensorflow.keras.preprocessing import image

# global variables
WIDTH = 128
HEIGHT = 128
SEARCH_LANG = ['en', 'ja']
COLORS = {'青': np.uint8([[[255, 0, 0]]]), '緑': np.uint8([[[0, 255, 0]]]),
            '赤': np.uint8([[[0, 0, 255]]]), '黄': np.uint8([[[0, 255, 255]]]),
            'ピンク': np.uint8([[[255, 0, 255]]]), '紫': np.uint8([[[255, 0, 127]]])}

def color(paths, c):
    ans = []
    for file_path in tqdm(paths):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        color_hsv = cv2.cvtColor(COLORS[c], cv2.COLOR_BGR2HSV)
        lower = np.array([color_hsv[0, 0, 0] - 10, 100, 100])
        upper = np.array([color_hsv[0, 0, 0] + 10, 255, 255])

        mask = cv2.inRange(img_hsv, lower, upper)

        retval = cv2.countNonZero(mask)
        percent = retval * 100 / (img.shape[0] * img.shape[1])

        if(percent > 0):
            ans.append(file_path)
    
    return ans

def shape(paths, s):
    ans = []
    for file_path in tqdm(paths):
        gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        gray = cv2.blur(gray,(9,9))

        ret, threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[1:]:
            # cv2.approxPloyDP() function to approximate the shape
            approx = cv2.approxPolyDP(
                contour, 0.01 * cv2.arcLength(contour, True), True)
            
            if len(approx) == 3 and s == '三角形':
                ans.append(file_path)
                break
        
            elif len(approx) == 4 and s == '四角形':
                ans.append(file_path)
                break

            elif len(approx) == 5 and s == '五角形':
                ans.append(file_path)
                break
                
            elif len(approx) == 6 and s == '六角形':
                ans.append(file_path)
                break
                
            else:
                if(s == '丸'):
                    ans.append(file_path)
                    break
    return ans

def text(paths, txt):
    reader = Reader(SEARCH_LANG, gpu = True)
    ans = []
    for file_path in tqdm(paths):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        results = reader.readtext(img)

        for (bbox, text, prob) in results:
            if text == txt:
                ans.append(file_path)
                break
    return ans

def cal_fid(source, target):
    img = [[source], [target]]
    fid_value = fid_mod.calculate_fid_given_paths(img)
    return fid_value

def cal_lpips(loss_fn_vgg, source, target):
    img0 = Image.open(source)
    img0 = img0.resize((WIDTH, HEIGHT))
    img0 = (T.to_tensor(img0) - 0.5) * 2
    img0.unsqueeze(0)

    img1 = Image.open(target)
    img1 = img1.resize((WIDTH, HEIGHT))
    img1 = (T.to_tensor(img1) - 0.5) * 2
    img1.unsqueeze(0)

    d = loss_fn_vgg(img0, img1)
    val = d.detach().numpy()
    return val[0, 0, 0, 0]

def fid_lpips(paths, target_path):
    loss_fn_vgg = lpips.LPIPS(net='vgg', lpips=True)
    ans = {}
    for file_path in tqdm(paths):
        # FID is slow, LPIPS is fast (and LPIPS seems better than FID)
        dist = cal_fid(file_path, target_path) + cal_lpips(loss_fn_vgg, target_path, file_path)
        # dist = cal_lpips(loss_fn_vgg, target_path, file_path)
        ans[file_path] = dist
    
    ans = {k: v for k, v in sorted(ans.items(), key=lambda item: item[1])}
    return list(ans.keys())
        
def target(paths, obj):
    detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite4/detection/1")
    labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
    labels = labels['OBJECT (2017 REL.)']
    ans = []
    for file_path in tqdm(paths):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rgb_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.uint8)
        rgb_tensor = tf.expand_dims(rgb_tensor , 0)

        boxes, scores, classes, num_detections = detector(rgb_tensor)

        pred_labels = classes.numpy().astype('int')[0] 
        pred_labels = [labels[i] for i in pred_labels]
        pred_scores = scores.numpy()[0]

        for score, label in zip(pred_scores, pred_labels):
            if score > 0.5 and label == obj:
                ans.append(file_path)
                break
    return ans

def deep_feature(paths, target_path):
    fe = FeatureExtractor()
    features = []
    for file_path in tqdm(paths):
        features.append(fe.extract(image.load_img(file_path)))
    
    features = np.array(features)
    query = np.array(fe.extract(image.load_img(target_path)))

    dists = np.linalg.norm(features-query, axis=1)
    ids = np.argsort(dists)

    ans = [paths[i] for i in ids]

    return ans

def compare_hist(paths, target_path):
    query = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    query_hist = cv2.calcHist([query],[0],None,[256],[0,256]) / len(query)

    ans = {}
    for file_path in tqdm(paths):
        gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        histograms = cv2.calcHist([gray],[0],None,[256],[0,256]) / len(gray)

        ans[file_path] = cv2.compareHist(query_hist, histograms, cv2.HISTCMP_CORREL)

    ans = {k: v for k, v in sorted(ans.items(), key=lambda item: item[1], reverse=True)}
    return list(ans.keys())

def knn(paths, features, query, show):
    dists = np.linalg.norm(features - query, axis=1)

    try:
        ids = np.argpartition(dists, show)[:show]

        # sort the dist (should be slower than the above)
        # ids = np.argsort(dists)
    except:
        ids = np.argpartition(dists, len(paths) - 1)[:len(paths)]

        # sort the dist (should be slower than the above)
        # ids = np.argsort(dists)
        
    # print(dists[ids])
    return [paths[i] for i in ids] 

def feature(paths, target_path, deep, show):
    reader = Reader(SEARCH_LANG, gpu = True)
    fe = FeatureExtractor()
    
    target = cv2.imread(target_path, cv2.IMREAD_COLOR)
    target = cv2.resize(target, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    target_blur = cv2.blur(target_gray,(9,9))

    query = []
    query.extend(get_color_feat(target))
    query.extend(get_shape_feat(target_blur))
    query.extend(get_hist_feat(target_path))

    results = reader.readtext(target)
    query_text = []
    if results:
        for (bbox, text, prob) in results:
            query.append(prob)
            query_text.append(text)

    if deep:
        query.extend(get_deep_feat(fe, target_path))
    
    query = np.array(query)

    features = []

    for index, file_path in enumerate(tqdm(paths)):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.blur(img_gray,(9,9))

        img_data = []

        img_data.extend(get_color_feat(img))
        img_data.extend(get_shape_feat(img_blur))
        img_data.extend(get_hist_feat(file_path))
        
        if(query_text):
            img_data.extend(get_text_feat(reader, img, query_text))
        
        if deep:
            img_data.extend(get_deep_feat(fe, file_path))

        features.append(img_data)
    features = np.array(features)

    ans = knn(paths, features, query, show)
    return ans

def get_deep_feat(fe, file_path):
    return fe.extract(image.load_img(file_path))

def get_hist_feat(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    histograms = cv2.calcHist([gray],[0],None,[256],[0,256]) / len(gray)

    return histograms.flatten()

def get_text_feat(reader, img, query_text):
    results = reader.readtext(img)

    tag = [0] * len(query_text)
    for (bbox, text, prob) in results:
        for i, q in enumerate(query_text):
            if(q == text):
                tag[i] = prob

    return tag

def get_color_feat(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    tag = [0] * len(COLORS)

    for index, color in enumerate(COLORS):
        color_hsv = cv2.cvtColor(COLORS[color], cv2.COLOR_BGR2HSV)
        lower = np.array([color_hsv[0, 0, 0] - 10, 100, 100])
        upper = np.array([color_hsv[0, 0, 0] + 10, 255, 255])

        mask = cv2.inRange(img_hsv, lower, upper)

        retval = cv2.countNonZero(mask)
        percent = retval * 100 / (img.shape[0] * img.shape[1])

        tag[index] = percent
    
    return tag

def get_shape_feat(gray):
    shape = ['三角形', '四角形', '五角形', '六角形', '丸']
    tag = {s: 0 for s in shape}

    ret, threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # list for storing names of shapes
    for contour in contours[1:]:
        # here we are ignoring first counter because 
        # findcontour function detects whole image as shape
    
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        
        if len(approx) == 3:
            tag['三角形'] += 1
    
        elif len(approx) == 4:
            tag['四角形'] += 1

        elif len(approx) == 5:
            tag['五角形'] += 1

        elif len(approx) == 6:
            tag['六角形'] += 1

        else:
            tag['丸'] += 1

    return list(tag.values())