#coding=utf-8
from osgeo import gdal
import numpy as np
from keras.utils.np_utils import to_categorical
import cv2
import random
import os
try:
    from cv2 import imread, imwrite
except ImportError:
    from skimage.io import imread, imsave
    imwrite = imsave
random.seed(7)

def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img,dtype="float")
    return img
def GDAL_load_img(path):
	dataset=gdal.Open(path)
	data=dataset.ReadAsArray()
	data=np.transpose(data,(1,2,0))
	return data
#save data to a tiff file
def writeTiff(im_data,im_width,im_height,im_bands,im_geotrans,im_proj,save_path):#
	if 'int8' in im_data.dtype.name:
		datatype = gdal.GDT_Byte
	elif 'int16' in im_data.dtype.name:
		datatype = gdal.GDT_UInt16
	else:
		datatype = gdal.GDT_Float32
	driver = gdal.GetDriverByName("GTiff")
	dataset = driver.Create(save_path, im_width, im_height, im_bands, datatype)
	if(dataset!= None):
		dataset.SetGeoTransform(im_geotrans)
		dataset.SetProjection(im_proj)
	if (im_bands==1):
		dataset.GetRasterBand(1).WriteArray(im_data)
	else:
		for i in range(im_bands):
			dataset.GetRasterBand(i+1).WriteArray(im_data[i])
	del dataset
def get_train_val(train_path,val_path):
	train_set = []
	val_set  = []
	for pic in os.listdir(os.path.join(train_path,'img')):
		train_set.append(pic)
	random.seed(0)
	random.shuffle(train_set)

	for pic1 in os.listdir(os.path.join(val_path,'img')):
		val_set.append(pic1)
	return train_set,val_set

# data for training(for multi class) 
def generateData_multl_class(filepath,batch_size,img_w,img_h,labelencoder,n_label,crop,iter,data=[]):  
	#print 'generateData...'
	while True:  
		train_data = []  
		train_label = []  
		batch = 0  
		for i in (range(len(data))): 
			url = data[i]
			batch += 1
			img = GDAL_load_img(os.path.join(filepath, 'img', url))
			if crop:
				img=img[0:img_w,0:img_h,:]
			train_data.append(img)
			label = load_img(os.path.join(filepath, 'seed_'+iter, url), grayscale=True)
			if crop:
				label=label[0:img_w,0:img_h]
			label = np.array(label).reshape((img_w * img_h,))
			train_label.append(label)
			if batch % batch_size==0: 
				train_data = np.array(train_data)  
				train_label = np.array(train_label).flatten()
				train_label = labelencoder.transform(train_label)
				train_label = to_categorical(train_label, num_classes=n_label+1)  
				train_label = train_label.reshape((batch_size,img_w * img_h,n_label+1))
				train_label=train_label[:,:,1:n_label+1]
				yield (train_data,train_label)
				train_data = []  
				train_label = []  
				batch = 0

# data for validation (for multi class) 
def generateValidData_multl_class(filepath,batch_size,img_w,img_h,labelencoder,n_label,crop,iter,data=[]):  
	#print 'generateValidData...'
	while True:  
		valid_data = []  
		valid_label = []  
		batch = 0  
		for i in (range(len(data))):
			url = data[i]
			batch += 1
			img = GDAL_load_img(os.path.join(filepath, 'img', url))
			if crop:
				img=img[0:img_w,0:img_h,:]
			valid_data.append(img)  
			label = load_img(os.path.join(filepath, 'seed_'+iter, url), grayscale=True)
			if crop:
				label=label[0:img_w,0:img_h]
			label = np.array(label).reshape((img_w * img_h,))
			valid_label.append(label)  
			if batch % batch_size==0:  
				valid_data = np.array(valid_data)  
				valid_label = np.array(valid_label).flatten()  
				valid_label = labelencoder.transform(valid_label)  
				valid_label = to_categorical(valid_label, num_classes=n_label+1)  
				valid_label = valid_label.reshape((batch_size,img_w * img_h,n_label+1))
				valid_label = valid_label[:,:,1:n_label+1]
				yield (valid_data,valid_label)  
				valid_data = []  
				valid_label = []  
				batch = 0