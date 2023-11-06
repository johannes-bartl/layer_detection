from UNET_model import UNet
from glob import glob
from training import load_images,DataGeneratorUNET
import matplotlib.pyplot as plt
from helper_functions import plot_grid
model = UNet(num_class=1,d=8,img_shape=(512,512,1))

#path to test data #TODO introduce test data set
gt_path = r"C:\Users\johan\Documents\1-Doktorarbeit\projects\layer_detection\data\training_data\gt"
img_path = r"C:\Users\johan\Documents\1-Doktorarbeit\projects\layer_detection\data\training_data\images"

images,gt = load_images(img_path,gt_path)
model.load_weights('weights.h5')



gen = DataGeneratorUNET(images,gt,512,512,16)
img,gt = gen[0]

prediction = model.predict(img)

fig,ax = plt.subplots(ncols=4,nrows=4)
for i,a in enumerate(ax.flatten()[::2]):
    a.imshow(img[i])
    ax.flatten()[i+1].imshow(gt[i])
plt.show()