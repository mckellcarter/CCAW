#!/usr/bin/env python
# encoding: utf-8
"""
sim_data.py

Created by McKell Carter on 2014-02-21.
Copyright (c) 2014 All rights reserved.
for ref see Ipython Notebook
"""


import sys, os
import numpy as N
import nibabel


#load a template file to use for header information
temp = nibabel.load('double_sphere.nii.gz')

#source 1
n1 = 4000
cvm1 = N.array([[1.0, 0.8],[0.8,1.]])
Ch_cvm1 = N.linalg.cholesky(cvm1)
x1 = N.random.normal(0,1,[n1,2])
x_cv1 = N.dot(Ch_cvm1, x1.T).T

#source 2
n2 = 1000
cvm2 = N.array([[1.0, 0.0],[0.0,1.]])
Ch_cvm2 = N.linalg.cholesky(cvm2)
x2 = N.random.normal(1,1,[n2,1])
x2 = N.hstack([x2,N.random.normal(2,1,[n2,1])])
x_cv2 = N.dot(Ch_cvm2, x2.T).T

#build matrices for output images
out_mat1 = N.zeros(temp.get_shape()).flatten()
out_mat1[0:len(x_cv1)] = x_cv1[:,0]
out_mat1[len(x_cv1):len(x_cv1)+len(x_cv2)] = x_cv2[:,0]
out_mat2 = N.zeros(temp.get_shape()).flatten()
out_mat2[0:len(x_cv1)] = x_cv1[:,1]
out_mat2[len(x_cv1):len(x_cv1)+len(x_cv2)] = x_cv2[:,1]
#build image objects
out_img1 = nibabel.Nifti1Image(out_mat1.reshape(temp.shape), affine=temp.get_affine(), header=temp.get_header())
out_img2 = nibabel.Nifti1Image(out_mat2.reshape(temp.shape), affine=temp.get_affine(), header=temp.get_header())
#save image objects
nibabel.save(out_img1, 'x_img.nii.gz')
nibabel.save(out_img2, 'y_img.nii.gz')
