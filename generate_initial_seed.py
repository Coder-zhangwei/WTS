import numpy as np
import os
from osgeo import gdal  
from sklearn.svm import SVC
from sklearn.externals import joblib
from tools import writeTiff
from tqdm import tqdm
import random
#generate the training data for SVM from txt files
#****txt_path----The path of txt files that save training data
#****norm_value---The normalized value
#****band_num--Band number of remote sensing images
def makeSVMTrainDataFromTxt(txt_path,norm_value,band_num):
	#print("Genetaing training data from txt files...")
	list =os.listdir(txt_path)
	new_data=np.zeros((0,band_num+1))
	for i in range(len(list)):
		if(os.path.splitext(list[i])[1]==".txt"):
			#print(list[i])
			roi_path=os.path.join(txt_path,list[i])
			a=np.loadtxt(roi_path)
			b=np.ones((a.shape[0],a.shape[1]+1))
			b[:,0:a.shape[1]]=a/norm_value
			b[:,a.shape[1]]=i+1
			new_data=np.vstack((new_data,b))
	return new_data
#tain svm using training data
def svm_train(txt_path,svm_save_path,norm_value,band_num):
	if not os.path.exists(svm_save_path):
		os.mkdir(svm_save_path)
	train_data=makeSVMTrainDataFromTxt(txt_path,norm_value,band_num)
	X=train_data[:,0:band_num]#spectral values
	Y=train_data[:,-1]#label
	print("SVM training......")
	clf=SVC(decision_function_shape='ovo',probability = True)
	clf.fit(X,Y)
	joblib.dump(clf, os.path.join(svm_save_path,"svm.m"))
#using SVM generate seed
def generate_seed_img(svm_save_path,train_img_path,seed_save_path,prob_threshold,norm_value):
	if not os.path.exists(seed_save_path):
		os.mkdir(seed_save_path)
	print("Generating seed......")
	clf = joblib.load(os.path.join(svm_save_path,"svm.m"))
	train_img_list =os.listdir(train_img_path)
	for train_img in train_img_list:
		if(os.path.splitext(train_img)[1]==".tif"):
			train_img_name=os.path.splitext(train_img)[0]
			dataset=gdal.Open(os.path.join(train_img_path,train_img))
			image=dataset.ReadAsArray()/norm_value
			im_geotrans = dataset.GetGeoTransform()
			im_proj = dataset.GetProjection()
			band_num,h,w = image.shape
			image1=image.reshape((band_num,(h*w)))
			image2=image1.T
			label_img=clf.predict(image2)
			label_img=label_img.reshape((h,w)).astype("uint8")
			prob_img=clf.predict_proba(image2)
			prob_img=prob_img.reshape((h,w,prob_img.shape[-1])).astype("float32")
			seed_position = (np.sum((prob_img> prob_threshold).astype(np.uint8),axis=2) > 0.5).astype(np.uint8)
			seed_img=label_img*seed_position
			writeTiff(seed_img,w,h,1,im_geotrans,im_proj,os.path.join(seed_save_path,train_img_name+"_seed.tif"))
#clip seed and corresponding imgs to gererate patches training data set
def generate_patches(img_path,seed_path,save_path,norm_value,img_size=256,stride=128):
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	if not os.path.exists(os.path.join(save_path,'img')):
		os.mkdir(os.path.join(save_path,'img'))
	if not os.path.exists(os.path.join(save_path,'seed_0')):
		os.mkdir(os.path.join(save_path,'seed_0'))
	print('Generating patches set...')
	list =os.listdir(img_path)
	g_count = 0
	#clip by sliding window(regularly)
	for i in tqdm(range(len(list))):
		img=os.path.join(img_path,list[i])
		src_img_dataset=gdal.Open(img)
		X_width=src_img_dataset.RasterXSize
		X_height=src_img_dataset.RasterYSize
		im_bands = src_img_dataset.RasterCount
		im_geotrans = src_img_dataset.GetGeoTransform()
		im_proj = src_img_dataset.GetProjection()
		seed_file_name = os.path.splitext(list[i])[0]+"_seed.tif"
		seed_dataset=gdal.Open(os.path.join(seed_path,seed_file_name))
		w=X_width - img_size
		h=X_height - img_size
		for regular_width in range(0,w,stride):
			for regular_height in range(0,h,stride):
				src_roi1= src_img_dataset.ReadAsArray(regular_width,regular_height,img_size,img_size)/norm_value
				label_roi= seed_dataset.ReadAsArray(regular_width,regular_height,img_size,img_size)
				save_img_name=os.path.join(save_path,'img')+'/%d_regular.tif' % g_count
				save_seed_name=os.path.join(save_path,'seed_0')+'/%d_regular.tif' % g_count
				writeTiff(src_roi1,img_size,img_size,im_bands,im_geotrans,im_proj,save_img_name)
				writeTiff(label_roi,img_size,img_size,1,im_geotrans,im_proj,save_seed_name)
				g_count += 1
				if g_count%1000==0:
					print('******clipping',g_count,'patches\n')
	#clip randomly
	for i in tqdm(range(len(list))):
		count=0
		img=os.path.join(img_path,list[i])
		src_img_dataset=gdal.Open(img)
		X_width=src_img_dataset.RasterXSize
		X_height=src_img_dataset.RasterYSize
		im_bands = src_img_dataset.RasterCount
		im_geotrans = src_img_dataset.GetGeoTransform()
		im_proj = src_img_dataset.GetProjection()
		seed_file_name = os.path.splitext(list[i])[0]+"_seed.tif"
		seed_dataset=gdal.Open(os.path.join(seed_path,seed_file_name))
		each_num=int((X_width*X_height)/(stride*stride))
		while count < each_num:
			random_width = random.randint(0, X_width - img_size - 1)
			random_height = random.randint(0, X_height - img_size - 1)
			src_roi1= src_img_dataset.ReadAsArray(random_width,random_height,img_size,img_size)/norm_value
			label_roi= seed_dataset.ReadAsArray(random_width,random_height,img_size,img_size)
			save_img_name=os.path.join(save_path,'img')+'/%d_random.tif' % g_count
			save_seed_name=os.path.join(save_path,'seed_0')+'/%d_random.tif' % g_count
			writeTiff(src_roi1,img_size,img_size,im_bands,im_geotrans,im_proj,save_img_name)
			writeTiff(label_roi,img_size,img_size,1,im_geotrans,im_proj,save_seed_name)
			count += 1 
			g_count += 1
			if g_count%1000==0:
				print('******clipping',g_count,'patches\n')
if __name__ == '__main__':
	training_data_txt_path="./data/points_training_data"#The path of txt files that save training data
	svm_save_path='./model/svm'# The path of SVM model

	#1.training SVM using points training data
	svm_train(training_data_txt_path,svm_save_path,17000.0,10)
	
	#2.generating seed using SVM
	train_img_path='./data/train/imgs'#The path of training images
	train_seed_save_path='./data/train/seed'#The path of training seed
	generate_seed_img(svm_save_path,train_img_path,train_seed_save_path,0.7,17000.0)
	val_img_path='./data/val/imgs'#The path of validation images
	val_seed_save_path='./data/val/seed'#The path of validation seed
	generate_seed_img(svm_save_path,val_img_path,val_seed_save_path,0.7,17000.0)
	
	#3.clipping images and corresponding seed into patches
	train_patches_path='./data/pathes_data/train'
	generate_patches(train_img_path,train_seed_save_path,train_patches_path,17000.0)
	val_patches_path='./data/pathes_data/val'
	generate_patches(val_img_path,val_seed_save_path,val_patches_path,17000.0)

