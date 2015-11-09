#!/usr/bin/env python
#Semi supervised novelty detction Method (based on Blanchard paper)

#Necessary packages
import math
import CCAW #local import
import sys,os  
import random as r
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score
from sklearn.neighbors.kde import KernelDensity

#Collecting the training data
x_image_pname = 'x_img_clean_train.nii.gz'
x_mask_pname =  x_image_pname  #self masked
y_image_pname = 'y_img_clean_train.nii.gz' 
y_mask_pname =  y_image_pname
X = CCAW.BaseData(x_image_pname, x_mask_pname)
Y = CCAW.BaseData(y_image_pname, y_mask_pname)	
X_mat = np.column_stack([X.image_data_masked, Y.image_data_masked])

kde1 = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X_mat)

x_image1_pname = 'x_img_mixed.nii.gz'
x_mask1_pname =  x_image1_pname  #self masked
y_image1_pname = 'y_img_mixed.nii.gz' 
y_mask1_pname =  y_image1_pname
X1 = CCAW.BaseData(x_image1_pname, x_mask1_pname)
Y1 = CCAW.BaseData(y_image1_pname, y_mask1_pname)	
X1_mat = np.column_stack([X1.image_data_masked, Y1.image_data_masked])

kde2 = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X1_mat)

#Collecting the test data (Nominal + Novel)
x_image1_pname = 'x_img_clean_test.nii.gz'
x_mask1_pname =  x_image1_pname  #self masked
y_image1_pname = 'y_img_clean_test.nii.gz' 
y_mask1_pname =  y_image1_pname
X1 = CCAW.BaseData(x_image1_pname, x_mask1_pname)
Y1 = CCAW.BaseData(y_image1_pname, y_mask1_pname)	
X1_mat = X1.image_data_masked
Y1_mat = Y1.image_data_masked

x_image2_pname = 'x_img_novel_test.nii.gz'
x_mask2_pname =  x_image2_pname  #self masked
y_image2_pname = 'y_img_novel_test.nii.gz' 
y_mask2_pname =  y_image2_pname
X2 = CCAW.BaseData(x_image1_pname, x_mask2_pname)
Y2 = CCAW.BaseData(y_image1_pname, y_mask2_pname)	
X2_mat = X2.image_data_masked
Y2_mat = Y2.image_data_masked

threshold = 0

X_test = []
Y_test = []
X_clean_plot =[]
Y_clean_plot = []
X_novel_plot =[]
Y_novel_plot = []
roc_true = []
roc_score = []

i = 0
for i in range(0,100000,1):
	index = r.randint(0,len(X1_mat)-1)
	X_test.append(X1_mat[index])
	Y_test.append(Y1_mat[index])
	X_clean_plot.append(X1_mat[index])
	Y_clean_plot.append(Y1_mat[index])
	roc_true.append(0)
j = 0
for j in range(0,100000,1):
	index = r.randint(0,len(X2_mat)-1)
	X_test.append(X2_mat[index])
	Y_test.append(Y2_mat[index])
	X_novel_plot.append(X2_mat[index])
	Y_novel_plot.append(Y2_mat[index])
	roc_true.append(1)

test_mat = np.column_stack([X_test, Y_test])

for item in test_mat:
	temp1 = kde2.score_samples(item)[0]
	temp2 = kde1.score_samples(item)[0]
	temp = float(temp1 / temp2)
	if temp > threshold :
		roc_score.append(1)
	else:
		roc_score.append(0)

print 'ROC Score: ', roc_auc_score(roc_true,roc_score)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X_clean_plot,Y_clean_plot,color ='green',s=5, edgecolor='none')
ax1.scatter(X_novel_plot,Y_novel_plot,color ='red',s=5, edgecolor='none')
ax1.set_aspect(1./ax1.get_data_ratio()) # make axes square
#plt.show()
plt.savefig('SSND_blan.png')
