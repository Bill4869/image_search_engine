import cv2
# import numpy as np
import matplotlib.pyplot as plt

def hist_diff():
    query_hist = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    train_hist = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    pt1 = kp1[m.queryIdx].pt
    size1 = int(kp1[m.queryIdx].size)
    pt2 = kp2[m.trainIdx].pt
    size2 = int(kp2[m.trainIdx].size)

    for i in range(3):
        count = 0
        for j in range(-1*(size1), (size1) + 1):
            for k in range(-1*(size1), (size1) + 1):
                try:
                    x = int(pt1[0])
                    y = int(pt1[1])
                    color = img1[y+j, x+k, i]
                    if(color < 64):
                        query_hist[i][0] += 1
                    elif(color < 128):
                        query_hist[i][1] += 1
                    elif(color < 192):
                        query_hist[i][2] += 1
                    else:
                        query_hist[i][3] += 1
                    count += 1
                except:
                    continue

        for l in range(4):
            query_hist[i][l] /= count
    
    for i in range(3):
        count = 0
        for j in range(-1*(size2), (size2) + 1):
            for k in range(-1*(size2), (size2) + 1):
                try:
                    x = int(pt2[0])
                    y = int(pt2[1])
                    color = img2[y+j, x+k, i]
                    if(color < 64):
                        train_hist[i][0] += 1
                    elif(color < 128):
                        train_hist[i][1] += 1
                    elif(color < 192):
                        train_hist[i][2] += 1
                    else:
                        train_hist[i][3] += 1
                    count += 1
                except:
                    continue
        for l in range(4):
            train_hist[i][l] /= count

    diff = 0
    for i in range(3):
        for j in range(4):
            diff += abs(query_hist[i][j] - train_hist[i][j])
    
    return diff

imgs = []
path = './dog/'
for i in range(1, 5):
    img = cv2.imread(path + '{}.jpg'.format(i), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    imgs.append(img)

sift = cv2.SIFT_create()

# brute-force matcher
bf = cv2.BFMatcher()
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# for m, n in matches:
#     print(kp1[m.trainIdx].pt, kp1[m.trainIdx].size)
#     print(m.trainIdx, n.trainIdx)
#     raise

count = 1
diff = 0
for i in range(len(imgs)):
    for j in range(i, len(imgs)):
        if(i == j):
            continue
        
        print('-----------------------------------------------')
        print('{} : {}'.format(i, j))
        img1 = imgs[i]
        img2 = imgs[j]

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(gray1,None)
        kp2, des2 = sift.detectAndCompute(gray2,None)

        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        total_diff = 0
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append([m])
                total_diff += hist_diff()
        
        print('kp1 : {}'.format(len(kp1)))
        print('kp2 : {}'.format(len(kp2)))
        print('match points : {}'.format(len(good)))
        print('diff : {}'.format(total_diff))
        # print("match : {}".format(len(good) * 2 / (len(kp1) + len(kp2)) * 100))
        img = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # cv2.imwrite('./output/{}.png'.format(str(count)), img)

        match_rate = len(good) * 2 / (len(kp1) + len(kp2)) * 100
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title('match(%):{:.3f} diff:{:.3f}'.format(match_rate, total_diff))
        plt.savefig('./output/{}.png'.format(str(count)))

        count += 1
