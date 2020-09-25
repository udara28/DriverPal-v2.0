#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:07:01 2020

@author: john
"""

import cv2 as cv
import numpy as np

# Set up dataset
num_frames = 438
path = "/home/john/Documents/Xilinx_Contest/DriverPal-v2.0/optical-flow/Test_Datasets/Kitti_2011_09_26_drive_0051/2011_09_26_drive_0051_sync/"
path_left = path + "rgb_left/data/"
path_right = path + "rgb_right/data/"

# Intrinsic parameters from calibration data
fx = 9.037596e+02 # pixels
fy = 9.019653e+02 # pixels
f = (fx + fy)/2
camera_distance = -3.875744e+02 # pixels, camera 1 -> camera 2

for frame_iter in range(0,num_frames):
    
    frame_iter_str = str(frame_iter).zfill(10)
    frame_l = cv.imread(path_left + frame_iter_str +".png")
    gray_l = cv.cvtColor(frame_l, cv.COLOR_BGR2GRAY)
    frame_r = cv.imread(path_right + frame_iter_str +".png")
    gray_r = cv.cvtColor(frame_r, cv.COLOR_BGR2GRAY)
    
    stereo = cv.StereoBM_create(numDisparities=96, blockSize=25)
    disparity = stereo.compute(gray_l,gray_r)
    # Output is [16,4] fixed-point. Convert to float
    disparity = np.array(disparity)
    disparity = disparity.astype(np.float32)/16 + 1
    
    # Calculating depth might be something like
    # For each bounding box
    #   Compute average disparity
    #   Depth = camera_distance/average disparity
    #   Convert from pixels to meters?
    
    depth_map = (disparity - stereo.getMinDisparity())/stereo.getNumDisparities()
    cv.imshow('Disparity Map', depth_map)
    if cv.waitKey(50) & 0xFF == ord('q'):
        break
    
cv.destroyAllWindows()