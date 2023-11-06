from UNET_model import UNet
from glob import glob
import imageio.v3 as iio
import numpy as np
from training import load_images,DataGeneratorUNET
import matplotlib.pyplot as plt
from helper_functions import plot_grid
import os
model = UNet(num_class=1,d=8,img_shape=(512,512,1))

output_path = 'data/training_data/prediction-2023-11-05-19-12-17'
model.load_weights('weights/2023-11-05-19-12-17-weights.h5')

data_path = 'data/training_data/images'
debug = False
for img in glob(os.path.join(data_path,'*.png')):
    _,filename = os.path.split(img)
    img = np.array(iio.imread(img),dtype=np.float32)
    img_w,img_h = img.shape
    H,W = 512,512
    overlap = 0.5
    bs = 16
    img_stack = []

    imgs_pad = np.zeros((img_w + int(W), img_h + int(H)))  # pad each image with W,H therefore W/2. and H/2. on the side
    imgs_pad[int(W / 2.):int(W / 2.) + img_w, int(H / 2.):int(H / 2.) + img_h] = img
    img_w, img_h = imgs_pad.shape  # define new image width/height

    if debug:
        plt.imshow(imgs_pad)
        plt.show()

    # number of cropped images in width/height axis

    N_w = int(np.ceil(img_w / (W * (1 - overlap))))
    N_h = int(np.ceil(img_h / (H * (1 - overlap))))
    print(f'#CROPS: N_w: {N_w}, N_h: {N_h}')
    stack = np.zeros((N_w * N_h , W, H))  # create empty image stack, then assign to handle crops on the border
    c = 0

    for x in np.arange(0, img_w, W * (1 - overlap), dtype='int'):
        for y in np.arange(0, img_h, H * (1 - overlap), dtype='int'):
            cache = imgs_pad[x:x + W, y:y + H]

            stack[c, 0:cache.shape[0], 0:cache.shape[1]] = cache
            c += 1

    if debug:
        ncols, nrows = N_h, N_w
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 8))
        for ax, i in zip(axes.flatten(), stack[:N_h * N_w]):
            ax.imshow(i / 255.)
        for ax in axes.flatten(): ax.xaxis.set_visible(False);ax.yaxis.set_visible(False);
        plt.show()

    pred = np.concatenate([model.predict(stack[batch * bs:batch * bs + bs, ...]) for batch in range(int(np.ceil(len(stack) / bs)))], axis=0)[...,0]
    print(f'Predicted {len(pred)} images')
    # if overlap is used cut all overlapping areas and only take middle part
    reduced_pred = pred[:,int(overlap / 2. * W): int(W - overlap / 2. * W), int(overlap / 2. * H): int(H - overlap / 2. * H)]

    if debug:
        ncols, nrows = N_h, N_w
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 8))
        for ax, i in zip(axes.flatten(), reduced_pred[:N_h * N_w]):
            ax.imshow(i)
        for ax in axes.flatten(): ax.xaxis.set_visible(False);ax.yaxis.set_visible(False);
        plt.show()

    images_pred = np.zeros((int(N_w * W * (1 - overlap)), int(N_h * H * (1 - overlap))))
    for counter, pr in enumerate(reduced_pred):
        j, k = (counter // N_h) % N_w, counter % N_h
        images_pred[int(j * W * (1 - overlap)):int(j * W * (1 - overlap) + W * (1 - overlap)),
        int(k * H * (1 - overlap)):int(k * H * (1 - overlap) + H * (1 - overlap))] = pr


    # self.images_pred = self.images_pred[:,int(W/4.):int(W/4.)+self.imgs.shape[1],int(H/4.):int(H/4.)+self.imgs.shape[2],:]
    padW = int(W / 2 - overlap / 2. * W)
    padH = int(H / 2 - overlap / 2. * H)
    images_pred = images_pred[padW:padW + img.shape[0], padH:padH + img.shape[1]]

    if debug:
        plt.imshow(images_pred)
        plt.show()

    from PIL import Image

    Image.fromarray((255*images_pred).astype(np.uint8)).save(os.path.join(output_path,f'pred-{filename}'))
    Image.fromarray(images_pred>0.4).save(os.path.join(output_path,f'predTH-{filename}'))
