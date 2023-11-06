#tensorflow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow_addons as tfa

#UNET implementation
class UNet(models.Model):
    def __init__(self, img_shape, num_class, d=32, weights=None):
        concat_axis = 3
        inputs = layers.Input(shape=img_shape)
        self.shape = img_shape

        conv1 = layers.Conv2D(d, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
        conv1 = layers.Conv2D(d, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = layers.Conv2D(d * 2, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(d * 2, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(d * 4, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(d * 4, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = layers.Conv2D(d * 8, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(d * 8, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = layers.Conv2D(d * 16, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = layers.Conv2D(d * 16, (3, 3), activation='relu', padding='same')(conv5)

        up_conv5 = layers.UpSampling2D(size=(2, 2))(conv5)
        ch, cw = self.get_crop_shape(conv4, up_conv5)
        crop_conv4 = layers.Cropping2D(cropping=(ch, cw))(conv4)
        up6 = layers.concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = layers.Conv2D(d * 8, (3, 3), activation='relu', padding='same')(up6)
        conv6 = layers.Conv2D(d * 8, (3, 3), activation='relu', padding='same')(conv6)

        up_conv6 = layers.UpSampling2D(size=(2, 2))(conv6)
        ch, cw = self.get_crop_shape(conv3, up_conv6)
        crop_conv3 = layers.Cropping2D(cropping=(ch, cw))(conv3)
        up7 = layers.concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = layers.Conv2D(d * 4, (3, 3), activation='relu', padding='same')(up7)
        conv7 = layers.Conv2D(d * 4, (3, 3), activation='relu', padding='same')(conv7)

        up_conv7 = layers.UpSampling2D(size=(2, 2))(conv7)
        ch, cw = self.get_crop_shape(conv2, up_conv7)
        crop_conv2 = layers.Cropping2D(cropping=(ch, cw))(conv2)
        up8 = layers.concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = layers.Conv2D(d * 2, (3, 3), activation='relu', padding='same')(up8)
        conv8 = layers.Conv2D(d * 2, (3, 3), activation='relu', padding='same')(conv8)

        up_conv8 = layers.UpSampling2D(size=(2, 2))(conv8)
        ch, cw = self.get_crop_shape(conv1, up_conv8)
        crop_conv1 = layers.Cropping2D(cropping=(ch, cw))(conv1)
        up9 = layers.concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = layers.Conv2D(d, (3, 3), activation='relu', padding='same')(up9)
        conv9 = layers.Conv2D(d, (3, 3), activation='relu', padding='same')(conv9)

        ch, cw = self.get_crop_shape(inputs, conv9)
        conv9 = layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        conv10 = layers.Conv2D(num_class, (1, 1), activation="sigmoid")(conv9)

        super().__init__(inputs=inputs, outputs=conv10)

        if weights is not None:
            self.load_weight_file(weights)

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2])
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1])
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    def load_weight_file(self, url):
        self.load_weights(str(url))

#loss UNET

from tensorflow.keras.backend import epsilon
#from keras.backend import epsilon
from tensorflow.python.keras.utils.losses_utils import reduce_weighted_loss, ReductionV2
def normal_focal_loss(gamma=2.0):

    def _normal_focal_loss(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.dtypes.cast(y_pred, dtype=tf.float32)
        y_true = tf.convert_to_tensor(y_true)
        y_true = tf.dtypes.cast(y_true, dtype=tf.bool)
        epsilon_ = tf.convert_to_tensor(epsilon(), tf.float32)
        p = y_pred
        q = 1 - p
        # avoid zeros in p and q --> would cause problems with log(0) later
        p = tf.math.maximum(p, epsilon_)
        q = tf.math.maximum(q, epsilon_)

        # Loss for the positive examples
        pos_loss = -(q ** gamma) * tf.math.log(p)
        # Loss for the negative examples
        neg_loss = -(p ** gamma) * tf.math.log(q)
        # choose either pos_loss or neg_loss, depending on the input label
        loss = tf.where(y_true, pos_loss, neg_loss)
        loss = reduce_weighted_loss(loss, reduction=ReductionV2.SUM_OVER_BATCH_SIZE)
        return loss

    return _normal_focal_loss