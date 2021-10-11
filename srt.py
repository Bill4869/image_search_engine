import cv2
import numpy as np
from easyocr import Reader
from tqdm import tqdm
import tensorflow_hub as hub
import pandas as pd
import tensorflow as tf

WIDTH = 128
HEIGHT = 128

SEARCH_LANG = ['en', 'ja']

def color(paths):
    colors = {'青': np.uint8([[[255, 0, 0]]]), '緑': np.uint8([[[0, 255, 0]]]),
            '赤': np.uint8([[[0, 0, 255]]]), '黄': np.uint8([[[0, 255, 255]]]),
            'ピンク': np.uint8([[[255, 0, 255]]]), '紫': np.uint8([[[255, 0, 127]]])}

    tag_val = []
    tag_index = {}
    ans = []
    for index, file_path in enumerate(tqdm(paths)):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        tags = {}
        cur = 0
        for color in colors:
            color_hsv = cv2.cvtColor(colors[color], cv2.COLOR_BGR2HSV)
            lower = np.array([color_hsv[0, 0, 0] - 10, 100, 100])
            upper = np.array([color_hsv[0, 0, 0] + 10, 255, 255])

            mask = cv2.inRange(img_hsv, lower, upper)

            retval = cv2.countNonZero(mask)
            percent = retval * 100 / (img.shape[0] * img.shape[1])

            if(percent > 0 and percent > cur):
                cur = percent
                tags = {color: cur}
        tag_val.append(tags)
        for tag in tags:
            if(tag in tag_index):
                tag_index[tag].append(index)
            else:
                tag_index[tag] = [index]

    for color in tag_index:
        c = {i:tag_val[i][color] for i in tag_index[color]}
        c = {k: v for k, v in sorted(c.items(), key=lambda item: item[1], reverse=True)}
        ans.extend(list(c.keys()))

    ans = [paths[i] for i in ans]
    ans.extend(list(set(paths) - set(ans)))
    return ans

def shape(paths):
    tag_val = []
    tag_index = {}
    ans = []
    for index, file_path in enumerate(tqdm(paths)):
        gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        gray = cv2.blur(gray,(9,9))

        ret, threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        tags = {}
        for contour in contours[1:]:
            # cv2.approxPloyDP() function to approximate the shape
            approx = cv2.approxPolyDP(
                contour, 0.01 * cv2.arcLength(contour, True), True)
            
            if len(approx) == 3:
                if('三角形' not in tags):
                    tags['三角形'] = 1
                else:
                    tags['三角形'] += 1
        
            elif len(approx) == 4:
                if('四辺形' not in tags):
                    tags['四辺形'] = 1
                else:
                    tags['四辺形'] += 1

            elif len(approx) == 5:
                if('五角形' not in tags):
                    tags['五角形'] = 1
                else:
                    tags['五角形'] += 1
                
            elif len(approx) == 6:
                if('六角形' not in tags):
                    tags['六角形'] = 1
                else:
                    tags['六角形'] += 1
                
            else:
                if('丸' not in tags):
                    tags['丸'] = 1
                else:
                    tags['丸'] += 1
        
        if(tags):
            max_key = max(tags, key= lambda x: tags[x])
            tags = {max_key: tags[max_key]}
        tag_val.append(tags)
        for tag in tags:
            if(tag in tag_index):
                tag_index[tag].append(index)
            else:
                tag_index[tag] = [index]
    
    for tag in tag_index:
        c = {i:tag_val[i][tag] for i in tag_index[tag]}
        c = {k: v for k, v in sorted(c.items(), key=lambda item: item[1], reverse=True)}
        ans.extend(list(c.keys()))

    ans = [paths[i] for i in ans]
    ans.extend(list(set(paths) - set(ans)))
    return ans

def text(paths):
    reader = Reader(SEARCH_LANG, gpu = True)
    
    tag_val = []
    tag_index = {}
    ans = []
    for index, file_path in enumerate(tqdm(paths)):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        results = reader.readtext(img)
        tags = {}
        for (bbox, text, prob) in results:
            tags[text] = prob
        
        if(tags):
            max_key = max(tags, key= lambda x: tags[x])
            tags = {max_key: tags[max_key]}
        tag_val.append(tags)
        for tag in tags:
            if(tag in tag_index):
                tag_index[tag].append(index)
            else:
                tag_index[tag] = [index]

    for tag in tag_index:
        c = {i:tag_val[i][tag] for i in tag_index[tag]}
        c = {k: v for k, v in sorted(c.items(), key=lambda item: item[1], reverse=True)}
        ans.extend(list(c.keys()))

    ans = [paths[i] for i in ans]
    ans.extend(list(set(paths) - set(ans)))
    return ans

def target(paths):
    detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite4/detection/1")
    labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
    labels = labels['OBJECT (2017 REL.)']
    
    tag_val = []
    tag_index = {}
    ans = []
    for index, file_path in enumerate(tqdm(paths)):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rgb_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.uint8)
        rgb_tensor = tf.expand_dims(rgb_tensor , 0)

        boxes, scores, classes, num_detections = detector(rgb_tensor)

        pred_labels = classes.numpy().astype('int')[0] 
        pred_labels = [labels[i] for i in pred_labels]
        pred_scores = scores.numpy()[0]

        tags = {}
        for score, label in zip(pred_scores, pred_labels):
            if score > 0.5:
                if(label not in tags):
                    tags[label] = score
                else:
                    if(score > tags[label]):
                        tags[label] = score
                
        if(tags):
            max_key = max(tags, key= lambda x: tags[x])
            tags = {max_key: tags[max_key]}
        tag_val.append(tags)
        for tag in tags:
            if(tag in tag_index):
                tag_index[tag].append(index)
            else:
                tag_index[tag] = [index]

    for tag in tag_index:
        c = {i:tag_val[i][tag] for i in tag_index[tag]}
        c = {k: v for k, v in sorted(c.items(), key=lambda item: item[1], reverse=True)}
        ans.extend(list(c.keys()))

    ans = [paths[i] for i in ans]
    ans.extend(list(set(paths) - set(ans)))
    return ans