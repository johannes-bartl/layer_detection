import clickpoints
import numpy as np
from UNET_model import UNet, normal_focal_loss
import os
import tensorflow as tf
from glob import glob
import imageio.v3 as iio
import matplotlib.pyplot as plt
import augmentation
from tensorflow.keras.optimizers import Adam

def plot_grid(img_batch,gt_batch):
    fig, ax = plt.subplots(ncols=4, nrows=4, )
    for a, img in zip(ax.flatten(), img_batch):
        a.imshow(img)
    plt.axis('off')
    plt.show()

    fig, ax = plt.subplots(ncols=4, nrows=4, )
    for a, gt in zip(ax.flatten(), gt_batch):
        a.imshow(gt)
    plt.axis('off')
    plt.show()

#
# #show available GPUS
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

def load_images(img_path,gt_path):
    gt,images = [],[]

    for p in glob(os.path.join(gt_path,'**\*.png'),recursive=True):
        basepath,filename = os.path.split(p)
        matching_img_path = glob(os.path.join(img_path,'**',filename),recursive=True)[0]
        if len(matching_img_path) != 0:
            gt_img = np.array(np.array(iio.imread(p),dtype=np.float32)>0,dtype=np.uint8)
            if len(gt_img.shape) == 3:
                gt_img = np.array(gt_img[...,0] > 0,dtype=np.uint8)
            gt.append(gt_img)
            img = np.array(iio.imread(matching_img_path),dtype=np.float32)
            if len(img.shape) == 3:
                img = img[...,0]
            images.append(img)
    return images,gt


# DataGenerator class UNET
class DataGeneratorUNET(tf.keras.utils.Sequence):
    def __init__(self, images, gt, W, H, batch_size=32, shuffle=True, augment=False):
        self.images = images
        self.gt = gt
        self.W = W
        self.H = H
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

    def __len__(self):
        n_batches = 16
        return n_batches

    def __getitem__(self, index, batch_size=16):

        images = self.images
        gt = self.gt
        W,H = self.W,self.H

        images_shape = [i.shape for i in images]
        img_ind = np.random.randint(0,len(images))

        x_random = np.random.randint(0,images_shape[img_ind][0]-W,size=(batch_size))
        y_random = np.random.randint(0,images_shape[img_ind][1]-H,size=(batch_size))

        img_batch = np.array([images[img_ind][x:x+W,y:y+H] for x,y in zip(x_random,y_random)])
        gt_batch = np.array([gt[img_ind][x:x+W,y:y+H] for x,y in zip(x_random,y_random)])

        # plot_grid(img_batch,gt_batch)

        if self.augment:
            # print('AUGMENT')
            img_batch, gt_batch = augmentation.flip(img_batch,gt_batch)
            # plot_grid(img_batch,gt_batch)

        return img_batch, gt_batch

if __name__ == "__main__":
    gt_path = r"C:\Users\johan\Documents\1-Doktorarbeit\projects\layer_detection\data\training_data\gt"
    img_path = r"C:\Users\johan\Documents\1-Doktorarbeit\projects\layer_detection\data\training_data\images"

    images,gt = load_images(img_path,gt_path)
    gen = DataGeneratorUNET(images,gt,512,512,batch_size=16,augment=True)
    img,g = gen[0]
    model = UNet(num_class=1,d=8,img_shape=(512,512,1))

    model.compile(loss=normal_focal_loss(gamma=2.0),optimizer=Adam(learning_rate=1e-4))

    model.fit(gen,epochs=10,batch_size=16)

    model.save_weights('weights.h5')


