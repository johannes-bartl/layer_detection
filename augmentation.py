import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
# Augmentation

def flip(images,images_gt,left_right=True,up_down=True):
    if left_right:
        if np.random.rand() > 0.5:
            images = images[:, ::-1, :]
            images_gt = images_gt[:, ::-1, :]
    if up_down:
        if np.random.rand() > 0.5:
            images = images[:,:,::-1]
            images_gt = images_gt[:,:,::-1]

    return images, images_gt
# class Augmentation():
#
#     def __init__(self, images, images_gt, augmentation_stack, W=512, H=512):
#
#         self.images = images
#         self.images_gt = images_gt
#         self.augmentation_stack = augmentation_stack
#         self.W, self.H = W, H
#
#     def rotate(self, low, high):
#         images, images_gt = self.images, self.images_gt
#         omega = np.random.uniform(low=low, high=high)
#         omega_rad = omega * np.pi / 180
#         images = tfa.image.rotate(images, omega_rad, interpolation='BILINEAR')
#         images_gt = tfa.image.rotate(images_gt[..., None], omega_rad, interpolation='BILINEAR')
#
#         self.images, self.images_gt, self.gt = images, images_gt[..., 0]
#         return images, images_gt
#
#     def flip(self, up_down=False, left_right=True):
#         images, images_gt = self.images, self.images_gt
#         if left_right:
#             images = images[:,::-1,:]
#             images_gt = images_gt[:,::-1,:]
#         # if up_down:
#         #     images = images[:,:,::-1]
#         #     images_gt = images_gt[:,:,::-1]
#
#         return images, images_gt
#
#     def contrast(self, low=0.3, high=2):
#         images, images_gt, gt = self.images, self.images_gt, self.gt
#         gt_ = gt.copy()
#         factor = np.random.uniform(low, high)
#
#         images = (images - np.mean(images)) * factor + np.mean(images)
#         images = np.where(images > 255., 255., images)
#         images = np.where(images < 0., 0., images)
#
#         self.images, self.images_gt, self.gt = images, images_gt, gt_
#         return images, images_gt, gt
#
#     def brightness(self, low=-50, high=50):
#         images, images_gt = self.images, self.images_gt
#         factor = int(np.random.uniform(low, high))
#         images += factor
#         images = np.where(images > 255., 255., images)
#         images = np.where(images < 0., 0., images)
#
#         self.images, self.images_gt= images, images_gt
#         return images, images_gt
#
#
#     def augment(self, plot=False):
#
#         for fct, arg in self.augmentation_stack.items():
#             if fct == 'rotate':
#                 self.rotate(*arg)
#             if fct == 'flip':
#                 self.flip(*arg)
#             if fct == 'contrast':
#                 self.contrast(*arg)
#             if fct == 'brightness':
#                 self.brightmess(*arg)