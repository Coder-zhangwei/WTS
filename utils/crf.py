import os
import re
import sys
import glob
import json
import time
import numpy as np 
#import skimage
#import skimage.io as imgio
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from utils.CC_labeling_8 import CC_lab
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool
from keras.models import load_model
from keras.utils.np_utils import to_categorical


def dense_crf(probs, img=None, class_num=5, n_iters=5,
              sxy_gaussian=(3, 3), compat_gaussian=3,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=4,
              srgb_bilateral=(5, 5, 5),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.

    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FU
LL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEF
ORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FUL
L_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFO
RE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).

    Returns:
      Refined predictions after MAP inference.
    """
    h, w, _ = probs.shape
    probs = probs.transpose(2, 0, 1).copy(order='C') # Need a contiguous array.
    d = dcrf.DenseCRF(w*h, class_num) # Define DenseCRF model.
    probs[probs==0.0]=1e-8
    U = -np.log(probs) # Unary potential.
    U = U.reshape((class_num, -1)) # Needs to be flat.
    U = U.astype(np.float32)
    d.setUnaryEnergy(U)
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral)
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((class_num, h, w)).transpose(1,2,0)
    return preds 
# get the classed in the seed
def get_seed_lasses(data,class_num=5):
	classes=[]
	for i in range(class_num):
		if np.sum(data==i+1)!=0:
			classes.append(i+1)
	return np.array(classes).astype('int64')
#generate new seed using SRG
def single_generate_seed_step(seed,prob,th_prob = 0.75,class_num=5):
	#make a tag
	tag = np.zeros((class_num))
	tag[get_seed_lasses(seed)-1] = 1.0
	tag = np.reshape(tag,[1,1,class_num])
	labels = [0]
	for i in range(class_num):
		labels.append(i+1)
	labelencoder = LabelEncoder()
	labelencoder.fit(labels)
	cue=np.array(seed).flatten()
	#print(cue)
	cue = labelencoder.transform(cue)
	cue = to_categorical(cue, num_classes=class_num+1)
	cue = np.reshape(cue,(seed.shape[0],seed.shape[1],class_num+1))
	cue = cue[:,:,1:class_num+1]
	
	existing_prob = prob*tag
	existing_prob_argmax = np.argmax(existing_prob, axis=-1) + 1 # to tell the background pixel and the not-satisfy-condition pixel
	existing_prob_fg_th_mask = (np.sum((existing_prob> th_prob).astype(np.uint8),axis=-1) > 0.5).astype(np.uint8) # if there is one existing category's score is bigger than th_f, the the mask is 1 for this pixel

	label_map = existing_prob_fg_th_mask*existing_prob_argmax
	# the label map is a two-dimensional map to show which category satisify the following three conditions for each pixel
	# 1. the category is in the tags of the image
	# 2. the category has a max probs among the tags
	# 3. the prob of the category is bigger that the threshold
	# and those three conditions is the similarity criteria
	# for the value in label_map, 0 is for no category satisifies the conditions, n is for the category n-1 satisifies the conditions
	cls_index = np.where(tag>0.5)[2] # the existing labels index
	for c in cls_index:
		mat = (label_map == (c+1))
		mat = mat.astype(int)
		cclab = CC_lab(mat)
		cclab.connectedComponentLabel() # this divide each connected region into a group, and update the value of cclab.labels which is a two-dimensional list to show the group index of each pixel
		high_confidence_set_label = set() # this variable colloects the connected region index
		for (x,y), value in np.ndenumerate(mat):
			if value == 1 and cue[x,y,c] == 1:
				high_confidence_set_label.add(cclab.labels[x][y])
			elif value == 1 and np.sum(cue[x,y,:]) == 1:
				cclab.labels[x][y] = -1
		for (x,y),value in np.ndenumerate(np.array(cclab.labels)):
			if value in high_confidence_set_label:
				cue[x,y,c] = 1
	new_cue=np.zeros((cue.shape[0],cue.shape[1],cue.shape[2]+1)).astype("int32")
	new_cue[:,:,1:]=cue
	new_cue=np.argmax(new_cue,axis=-1)
	return new_cue