#coding=utf-8
from keras.models import Model
from keras.layers import Reshape,Input
from keras import backend as K
import segmentation_models as sm
K.set_image_data_format('channels_last')

def U_Net(back_bone,img_w, img_h, input_band_num, n_class):
    base_model = sm.Unet(back_bone, classes=n_class, activation='softmax', input_shape=(img_w, img_h, input_band_num), encoder_weights=None)
    output1=base_model.output
    output=Reshape((img_w*img_h,n_class))(output1)
    model = Model(base_model.input,output)
    model.compile(loss=seeded_loss,optimizer='Adam',metrics=[seeded_loss])
    return model

#seeded_loss
def seeded_loss(y_true,y_pred):
    count = K.sum(y_true)
    return -K.sum(y_true*K.log(y_pred+1e-8))/count




