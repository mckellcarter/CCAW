#!/usr/bin/env python
# encoding: utf-8
"""
CCAW_example.py

Created by McKell Carter on 2014-02-21.
Copyright (c) 2014. All rights reserved.
for devel. ref. see Ipython Notebook
"""


import sys, os
import numpy as N

import CCAW #local import

x_image_pname = 'x_img.nii.gz'
x_mask_pname =  x_image_pname  #self masked
y_image_pname = 'y_img.nii.gz' 
y_mask_pname =  y_image_pname
X = CCAW.BaseData(x_image_pname, x_mask_pname)
Y = CCAW.BaseData(y_image_pname, y_mask_pname)	
A = CCAW.Analysis3D(X, Y, 'contrast_density_test.png')

#Gaussian mixtures clustering
from sklearn import mixture
X_mat = N.hstack([X.image_data_masked, Y.image_data_masked])
dpgmm = mixture.DPGMM(alpha=2.5, n_components=10, covariance_type='full')
dpgmm.fit(X_mat)


