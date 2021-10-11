import glob, os, time
import cv2
import numpy as np

import search, srt

def get_img(path):
    # filenames = glob.glob(path + "*.jpg")
    file_paths = []
    for folder, sub_folders, files in os.walk(path):
        for special_file in files:
            if '.jpg' in special_file.lower():
                file_paths.append(os.path.join(folder, special_file))
    
    return file_paths

def display(paths):
    blank_image = np.zeros((128,128,3), np.uint8)

    l = len(paths) // 4 + (len(paths) % 4 > 0)

    im_list = []
    for i in range(l):
        im = []
        for j in range(4):
            try:
                img = cv2.imread(paths.pop(0), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                im.append(img)
            except:
                im.append(blank_image)

        im_list.append(im)
    
    im_title = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list])

    cv2.imwrite('result.jpg', im_title)

def find(path, color, shape, obj, text, image, deep, show):
    file_paths = get_img(path)

    start = time.time()
    if(shape and file_paths):
        file_paths = search.shape(file_paths, shape)
    
    if(color and file_paths):
        file_paths = search.color(file_paths, color)


    if(text and file_paths):
        file_paths = search.text(file_paths, text)
    
    if(obj and file_paths):
        file_paths = search.target(file_paths, obj)

    if(image and file_paths):
        file_paths = search.feature(file_paths, image, deep, show)

        # compute similarity based on FID and LPIPS
        # file_paths = search.fid_lpips(file_paths, image)

        # compute similarity based on color histogram comparison
        # file_paths = search.compare_hist(file_paths, image)

        # compute similarity based on deep features
        # file_paths = search.deep_feature(file_paths, image)

    print('time: {}'.format(time.time() - start))
    try:
        display(file_paths[:show])
    except:
        display(file_paths)


def sort(path, by, show):
    file_paths = get_img(path)

    start = time.time()

    if(by == 'color'):
        file_paths = srt.color(file_paths)
    elif(by == 'shape'):
        file_paths = srt.shape(file_paths)
    elif(by == 'text'):
        file_paths = srt.text(file_paths)
    elif(by == 'object'):
        file_paths = srt.target(file_paths)

    print('time: {}'.format(time.time() - start))
    try:
        display(file_paths[:show])
    except:
        display(file_paths)


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path", type=str, default='../images/mix', help='Path to images')
    # search options
    parser.add_argument("--search", action="store_true", help='searching function')
    parser.add_argument("-c", "--color", type=str, default=None, help='searching color [赤、青、緑、黄、ピンク、紫]')
    parser.add_argument("-s", "--shape", type=str, default=None, help='searching shape [三角形、読ん角形、五角形、六角形、丸]')
    parser.add_argument("-o", "--object", type=str, default=None, help='searching object')
    parser.add_argument("-t", "--text", type=str, default=None, help='searching text')
    
    parser.add_argument("-i", "--image", type=str, default=None, help='searching image')
    parser.add_argument("--deep", action="store_true", help='use deep feature in finding similar images')
    parser.add_argument("--show", type=int, default=10, help='number of images to show')
    # sort options
    parser.add_argument("--sort", action="store_true", help='sorting function')
    parser.add_argument("-b", "--by", type=str, default=None, help='type to sort [color, shape, object, text]')

    args = parser.parse_args()

    if args.search:
        find(args.path, args.color, args.shape, args.object, args.text, args.image, args.deep, args.show)
    elif args.sort:
        sort(args.path, args.by, args.show)