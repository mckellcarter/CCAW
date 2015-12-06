#!/usr/bin/env python
#Inductive Method (based on Blanchard paper)

#Necessary packages
import math
import CCAW
import sys,os  
import random as r
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score
from sklearn.neighbors.kde import KernelDensity

threshold = 0
try:
	threshold = float(sys.argv[1])
except IndexError:
	threshold = 6.07099938798e-13
print 'Threshold', threshold

#Collecting the training data
x_image_pname = 'x_img_clean_train.nii.gz'
x_mask_pname =  x_image_pname  #self masked
y_image_pname = 'y_img_clean_train.nii.gz' 
y_mask_pname =  y_image_pname
X = CCAW.BaseData(x_image_pname, x_mask_pname)
Y = CCAW.BaseData(y_image_pname, y_mask_pname)	
X_mat = np.column_stack([X.image_data_masked, Y.image_data_masked])



kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X_mat)


#Collecting the test data (Nominal + Novel)
x_image1_pname = 'x_img_clean_test.nii.gz'
x_mask1_pname =  x_image1_pname  #self masked
y_image1_pname = 'y_img_clean_test.nii.gz' 
y_mask1_pname =  y_image1_pname
X1 = CCAW.BaseData(x_image1_pname, x_mask1_pname)
Y1 = CCAW.BaseData(y_image1_pname, y_mask1_pname)	
X1_mat = X1.image_data_masked
Y1_mat = Y1.image_data_masked

x_image2_pname = 'x_img_novel.nii.gz'
x_mask2_pname =  x_image2_pname  #self masked
y_image2_pname = 'y_img_novel.nii.gz' 
y_mask2_pname =  y_image2_pname
X2 = CCAW.BaseData(x_image2_pname, x_mask2_pname)
Y2 = CCAW.BaseData(y_image2_pname, y_mask2_pname)	
X2_mat = X2.image_data_masked
Y2_mat = Y2.image_data_masked


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
	index = np.random.randint(0,len(X1_mat)-1)
	X_test.append(X1_mat[index])
	Y_test.append(Y1_mat[index])
	roc_true.append(0)
j = 0
for j in range(0,100000,1):
	index = np.random.randint(0,len(X2_mat)-1)
	X_test.append(X2_mat[index])
	Y_test.append(Y2_mat[index])
	roc_true.append(1)

test_mat = np.column_stack([X_test, Y_test])

count = 0
i= 0
for item in test_mat:
	temp = kde.score_samples(item)[0]
	temp = float(math.exp(temp))
	if temp < threshold :
		if roc_true[i] == 0:
			count = count + 1
			print count, '1:', roc_true[i], temp
		roc_score.append(1)
		X_novel_plot.append(item[0])
		Y_novel_plot.append(item[1])
	else:
		if roc_true[i] == 1:
			count = count + 1
			print count, '0:', roc_true[i], temp 
		roc_score.append(0)
		X_clean_plot.append(item[0])
		Y_clean_plot.append(item[1])
	i+=1

print 'ROC Score: ', roc_auc_score(roc_true,roc_score)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X_clean_plot,Y_clean_plot,color ='green',s=5, edgecolor='none')
ax1.scatter(X_novel_plot,Y_novel_plot,color ='red',s=5, edgecolor='none')
ax1.set_aspect(1./ax1.get_data_ratio()) # make axes square
#plt.show()
plt.savefig('Inductive_blan.png')
print 'Saving the file as Inductive_blan.png'



