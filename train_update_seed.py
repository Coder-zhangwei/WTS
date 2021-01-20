#coding=utf-8
from osgeo import gdal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from sklearn.preprocessing import LabelEncoder
from tools import generateData_multl_class,generateValidData_multl_class,get_train_val,writeTiff,load_img
import os
from model import U_Net,seeded_loss
from keras.models import load_model
from utils.crf import single_generate_seed_step,dense_crf

K.set_image_data_format('channels_last')
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(7)

#updateing seed using fullly connected CRF and SRG
def updatate_seed(new_model_path,patch_img_path,last_seed_path,new_seed_path,prob_threshold,img_size=256,class_num=5):
	#load unet model
	model = load_model(new_model_path,custom_objects={'seeded_loss': seeded_loss})
	patch_list =os.listdir(patch_img_path)
	for patch in patch_list:
		#open patch image
		src_img_dataset=gdal.Open(os.path.join(patch_img_path, patch))
		X_width=src_img_dataset.RasterXSize
		X_height=src_img_dataset.RasterYSize
		im_geotrans = src_img_dataset.GetGeoTransform()
		im_proj = src_img_dataset.GetProjection()
		img=src_img_dataset.ReadAsArray()
		img=np.transpose(img,(1,2,0))
		img1 = np.expand_dims(img, axis=0)
		#predict prob maps
		pred = model.predict(img1,verbose=2)  
		pred = pred.reshape((img_size,img_size,class_num)).astype("float")
		#using CRF optimize pred
		crf_result=dense_crf(pred,img)
		#open the last seed
		last_seed= load_img(os.path.join(last_seed_path, patch), grayscale=True)
		#using SRG generate new seed based on the crf_result
		new_seed=single_generate_seed_step(last_seed,crf_result,prob_threshold)
		#save new seed to tiff
		writeTiff(np.int8(new_seed),X_width,X_height,1,im_geotrans,im_proj,os.path.join(new_seed_path, patch))

def train_update_seed(args):
	class_num = 5#number of classes
	classes = [0., 1. , 2. , 3. , 4., 5.]#0stands for no seed point
	labelencoder = LabelEncoder()
	labelencoder.fit(classes)
	input_band_num=10#band number of input images
	img_size=256#image size 
	EPOCHS = 10
	BS = 5
	iter=10#number of iteration
	if not os.path.exists(args['data_train']):
		os.mkdir(args['data_train'])
	if not os.path.exists(args['data_val']):
		os.mkdir(args['data_val'])
	if not os.path.exists(args['model']):
		os.mkdir(args['model'])
	train_set_img_path=os.path.join(args['data_train'],"img")
	val_set_img_path=os.path.join(args['data_val'],"img")
	for i in range(iter):
		print("*****The ",i," iteration*****")
		model=U_Net('resnet50',img_size, img_size, input_band_num, class_num)
		model_save_path=os.path.join(args['model'],str(i)+'.h5')
		modelcheck = ModelCheckpoint(model_save_path,monitor='val_loss',save_best_only=True,mode='min')
		log = CSVLogger(os.path.join(args['model'],str(i)+'.csv'))
		callable = [modelcheck, log]
		train_set,val_set = get_train_val(args['data_train'],args['data_val'])
		train_numb = len(train_set)  
		valid_numb = len(val_set)  
		print ("The number of train data is",train_numb)
		print ("The number of val data is",valid_numb)
		H = model.fit_generator(generator=generateData_multl_class(args['data_train'],BS,img_size,img_size,labelencoder,class_num,False,str(i),train_set),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,validation_data=generateValidData_multl_class(args['data_val'],BS,img_size,img_size,labelencoder,class_num,False,str(i),val_set),validation_steps=valid_numb//BS,callbacks=callable)
		plt.style.use("ggplot")
		plt.figure()
		N = EPOCHS
		plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
		plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
		plt.title("Training Loss and Accuracy on Unet Satellite Seg")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss")
		plt.legend(loc="lower left")
		plt.savefig(os.path.join(args['model'],str(i)+'.png'))

		print("Update seed of the ",i," iteration")
		old_train_seed_path=os.path.join(args['data_train'],"seed_"+str(i))
		new_train_seed_path=os.path.join(args['data_train'],"seed_"+str(i+1))
		folder = os.path.exists(new_train_seed_path)
		if not folder:
			os.makedirs(new_train_seed_path)
		old_val_seed_path=os.path.join(args['data_val'],"seed_"+str(i))
		new_val_seed_path=os.path.join(args['data_val'],"seed_"+str(i+1))
		folder = os.path.exists(new_val_seed_path)
		if not folder:
			os.makedirs(new_val_seed_path) 
		updatate_seed(model_save_path,train_set_img_path,old_train_seed_path,new_train_seed_path,0.95+i*0.002)
		updatate_seed(model_save_path,val_set_img_path,old_val_seed_path,new_val_seed_path,0.95+i*0.002)
def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data_train", help="training data's path",
                    default='./data/pathes_data/train/')#path of training data
    ap.add_argument("-v", "--data_val", help="validaton data's path",
                    default='./data/pathes_data/val/')#path of validation data
    ap.add_argument("-m", "--model", default='./model/unet/',
                    help="path to output model")#path that saves model
    args = vars(ap.parse_args()) 
    return args
if __name__=='__main__':
    args = args_parse()
	#training u-net and update seed ieratively
    train_update_seed(args)