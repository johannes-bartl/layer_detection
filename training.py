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
from datetime import datetime
from helper_functions import parse_cfg, plot_grid

def load_images(img_path,gt_path):
    gt,images = [],[]

    #search recurively for all pngs in the gt folder
    for p in glob(os.path.join(gt_path,'**\*.png'),recursive=True):
        basepath,filename = os.path.split(p) #extract the filename
        # search for the filename in the image folder
        matching_img_path = glob(os.path.join(img_path,'**',filename),recursive=True)[0]
        if len(matching_img_path) != 0: #if it was found add both to the dataset
            gt_img = np.array(np.array(iio.imread(p),dtype=np.float32)>0,dtype=np.uint8)
            #if shape: (W,H,3) then reduce to (W,H)
            if len(gt_img.shape) == 3:
                gt_img = np.array(gt_img[...,0] > 0,dtype=np.uint8)
            gt.append(gt_img)
            img = np.array(iio.imread(matching_img_path),dtype=np.float32)
            if len(img.shape) == 3:
                img = img[...,0]
            images.append(img)
        else:
            print(f'{filename} not found in the image folder')
    return images,gt


# DataGenerator class UNET
class DataGeneratorUNET(tf.keras.utils.Sequence):

    def __init__(self, images, gt, W, H, batch_size=32, augment=False):
        self.images = images
        self.gt = gt
        self.W = W
        self.H = H
        self.batch_size = batch_size
        # self.shuffle = shuffle
        self.augment = augment

    def __len__(self):
        return len(self.images)

    # this is called whenever gen[i] is applied. The fit function calls this repeatedly during training
    def __getitem__(self, index, batch_size=16):

        images = self.images
        gt = self.gt
        W,H = self.W,self.H

        images_shape = [i.shape for i in images]
        img_ind = np.random.randint(0,len(images)) #for this batch choose one random image

        #find n (n=batch_size) random x and y position inside the image
        x_random = np.random.randint(0,images_shape[img_ind][0]-W,size=(batch_size))
        y_random = np.random.randint(0,images_shape[img_ind][1]-H,size=(batch_size))

        #and take the crops
        img_batch = np.array([images[img_ind][x:x+W,y:y+H] for x,y in zip(x_random,y_random)])
        gt_batch = np.array([gt[img_ind][x:x+W,y:y+H] for x,y in zip(x_random,y_random)])

        # plot_grid(img_batch,gt_batch)

        #augment to increase amount of available data in the training set
        if self.augment:
            # print('AUGMENT')
            img_batch, gt_batch = augmentation.flip(img_batch,gt_batch)
            # plot_grid(img_batch,gt_batch)
            #TODO add rotations?

        return img_batch, gt_batch #return shape (batch_size,W,H)

if __name__ == "__main__":

    cfg = parse_cfg('config.yaml')
    H,W = cfg["images"]["H"],cfg["images"]["W"]

    # show available GPUS & set them up
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    #the programm will take all pngs from the gt_path (recursively) and find the corresponding image in the img_path
    images,gt = load_images(cfg["path"]["img"],cfg["path"]["gt"])

    #TODO write function to split data into train, validation and test set

    #setup DataGenerator
    gen = DataGeneratorUNET(images,gt,H=H,W=W,batch_size=cfg["training"]["batch_size"],augment=True)
    img,gt = gen[0]

    #TODO setup DataGenerator on validation set without augmentation
    # gen_val = DataGeneratorUNET(images_val,gt_val,H=H,W=W,batch_size=batch_size,augment=False)

    #check if augmentation is correct: Does the image match the gt?
    plot_grid(img,gt)

    #setup UNet, define the img_shape (has to have one extra dimension in the end e.g. (512,512,1))
    #d defined the number of kernels for each convolutional layer: to see the network structure: print(model.summary)
    model = UNet(num_class=1,d=cfg["training"]["d"],img_shape=(W,H,1))

    #load weights into network to continue training
    if cfg["path"]["weight"] != None:
        model.load_weights(cfg["path"]["weight"])

    #compiling the network: using normal_focal_loss (#TODO test out different loss functions, maybe DICE)
    #normal focal loss can also handle class inaccuracies
    model.compile(loss=normal_focal_loss(gamma=2.0),optimizer=Adam(learning_rate=cfg["training"]["learning_rate"]))


    model.fit(gen,epochs=cfg["training"]["epochs"],batch_size=cfg["training"]["batch_size"])

    #after training save_weights
    model.save_weights(os.path.join(cfg["path"]["output"],f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'))