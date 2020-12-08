#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 13:26:48 2020

@author: john
"""
import cv2 as cv
import numpy as np

# Set up dataset
num_frames = 438
path = "/home/john/Documents/Xilinx_Contest/DriverPal-v2.0/estimateDistVel/Test_Datasets/Kitti_2011_09_26_drive_0051/2011_09_26_drive_0051_sync/"
path_left = path + "rgb_left/data/"
path_right = path + "rgb_right/data/"

# Intrinsic parameters from calibration data
f = 721.5377 # pixels, from EvaluationUtils
camera_distance = 0.54 # meters, from EvaluationUtils. Camera 2 -> 3

for frame_iter in range(0,num_frames):
    
    frame_iter_str = str(frame_iter).zfill(10)
    frame_l = cv.imread(path_left + frame_iter_str +".png")
    #gray_l = cv.cvtColor(frame_l, cv.COLOR_BGR2GRAY)
    frame_r = cv.imread(path_right + frame_iter_str +".png")
    #gray_r = cv.cvtColor(frame_r, cv.COLOR_BGR2GRAY)
    
    window_size = 5
    min_disp = 0
    num_disp = 96
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    disp = stereo.compute(frame_l, frame_r).astype(np.float32) / 16.0
    
    # Calculating depth might be something like
    # For each bounding box
    #   Compute average disparity
    #   Depth = camera_distance/average disparity
    #   Convert from pixels to meters?
    
    depth_map = (disp - min_disp)/num_disp
    cv.imshow('Disparity Map', depth_map)
    cv.imshow('Video Feed', frame_l)
    if cv.waitKey(50) & 0xFF == ord('q'):
        break
    
cv.destroyAllWindows()