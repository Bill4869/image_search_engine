import lpips
# import torch
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as T
# import cv2
import glob
from PIL import Image 

img_path = "./dog/"
filenames = glob.glob(img_path + "*.jpg")
filenames.sort()

# loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

# path = './results'
# path2 = './example'

# ds = ImageFolder(path, transform=T.Compose([T.ToTensor(), T.Resize((128, 128))]))

for i in range(len(filenames)):
    for j in range(i+1, len(filenames)):
        img0 = Image.open(filenames[i])
        img0 = img0.resize((128, 128))
        img0 = (T.to_tensor(img0) - 0.5) * 2
        img0.unsqueeze(0)

        img1 = Image.open(filenames[j])
        img1 = img1.resize((128, 128))
        img1 = (T.to_tensor(img1) - 0.5) * 2
        img1.unsqueeze(0)

        d = loss_fn_vgg(img0, img1)
        val = d.detach().numpy()
        print(val[0, 0, 0, 0])
        print(filenames[i], filenames[j])
        print('LPIPS: {}'.format(val))


# ds2 = ImageFolder(path2, transform=T.Compose([T.ToTensor(), T.Resize((128, 128))]))
# ds = ImageFolder(path)

# dl = DataLoader(ds, 1, num_workers=2, pin_memory=True)

# image, _ = ds[0]
# print(image.size)

# img0 = cv2.imread('./results/000000022969.png')
# print(type(img0))
# img1 = cv2.imread('./results/000000022969.png')

# img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
# img1 = torch.zeros(1,3,64,64)


# for img, img2 in zip(ds, ds2):
#     d = loss_fn_vgg(img[0], img2[0])

#     print(d)