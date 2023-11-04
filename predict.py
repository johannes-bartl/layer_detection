from UNET_model import UNet
from training import load_images,DataGeneratorUNET
import matplotlib.pyplot as plt
model = UNet(num_class=1,d=8,img_shape=(512,512,1))

gt_path = r"C:\Users\johan\Documents\1-Doktorarbeit\projects\layer_detection\data\training_data\gt"
img_path = r"C:\Users\johan\Documents\1-Doktorarbeit\projects\layer_detection\data\training_data\images"

images,gt = load_images(img_path,gt_path)
model.load_weights('weights.h5')



gen = DataGeneratorUNET(images,gt,512,512,32)
img,gt = gen[0]

prediction = model.predict(img)

print(prediction.shape)
n = 1
fig,ax = plt.subplots(ncols=3)
ax[0].imshow(img[n])
ax[1].imshow(prediction[n])
ax[2].imshow(gt[n])
plt.show()