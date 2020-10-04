#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 15:10:34 2020

@author: john
"""

import cv2 as cv
import numpy as np

# Read first two frames
num_frames = 438
path = "/home/john/Documents/Xilinx_Contest/DriverPal-v2.0/estimateDistVel/Test_Datasets/Kitti_2011_09_26_drive_0051/2011_09_26_drive_0051_sync/rgb_left/data/"

frame_iter = 0
frame_iter_str = str(frame_iter).zfill(10)
prev_frame = cv.imread(path + frame_iter_str +".png")
prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

frame_iter += 1
frame_iter_str = str(frame_iter).zfill(10)
frame = cv.imread(path + frame_iter_str +".png")
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# Set tracking visuals
mask = np.zeros_like(prev_frame)
color = (0,0,255)

# Detect FAST features
min_features = 2200
fast_threshold = 50
nonMaxSupression = True
fast = cv.FastFeatureDetector_create(fast_threshold,nonMaxSupression)
prev_kp = cv.KeyPoint_convert(fast.detect(prev_gray,None))

# Match features
lk_params = dict(winSize = (21,21), maxLevel = 3, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01), flags = 0, minEigThreshold = 0.001)
next_kp, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev_kp, None, **lk_params)

prev_gray = gray.copy()
prev_kp = next_kp.copy()

j = 0

for k in range(2,num_frames):
    # Read frame
    frame_iter += 1
    frame_iter_str = str(frame_iter).zfill(10)
    frame = cv.imread(path + frame_iter_str +".png")
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Match features in next image
    next_kp, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev_kp, None, **lk_params)
    
    # Throw out bad features
    next_kp = np.expand_dims(next_kp, 1)
    next_kp = next_kp[status == 1]
    prev_kp = np.expand_dims(prev_kp, 1)
    prev_kp = prev_kp[status == 1]
    
    # Throw out features that went out of frame
    next_kp_good_mask = np.ones(len(next_kp),dtype=bool)
    for k in range(len(next_kp)):
        if next_kp[k,0] < 0 or next_kp[k,1] < 0:
            next_kp_good_mask[k] = 0
    next_kp = next_kp[next_kp_good_mask == 1]
    prev_kp = prev_kp[next_kp_good_mask == 1]
    
    # If not enough features can be found, run detection again
    if len(prev_kp) < min_features:
        j += 1
        prev_kp = cv.KeyPoint_convert(fast.detect(prev_gray,None))
        # Match features in next image
        lk_params = dict(winSize = (21,21), maxLevel = 3, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
        next_kp, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev_kp, None, **lk_params)
        
        # Throw out bad features
        next_kp = np.expand_dims(next_kp, 1)
        next_kp = next_kp[status == 1]
        prev_kp = np.expand_dims(prev_kp, 1)
        prev_kp = prev_kp[status == 1]
        
        # Throw out features that went out of frame
        next_kp_good_mask = np.ones(len(next_kp),dtype=bool)
        for k in range(len(next_kp)):
            if next_kp[k,0] < 0 or next_kp[k,1] < 0:
                next_kp_good_mask[k] = 0
        next_kp = next_kp[next_kp_good_mask == 1]
        prev_kp = prev_kp[next_kp_good_mask == 1]
    
    
    #prev_frame_kp = cv.drawKeypoints(prev_frame, prev_kp, None, color=(0,0,255))
    #cv.imshow('img keypoints', prev_frame_kp)
    
    # Draw lines to matching features from previous frame
    mask = np.zeros_like(frame)
    for i, (new, old) in enumerate(zip(next_kp, prev_kp)):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        a, b = new.ravel()
        # Returns a contiguous flattened array as (x, y) coordinates for old point
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), color, 2)
        frame = cv.circle(frame, (a, b), 3, color, -1)
        # Overlays the optical flow tracks on the original frame
        output = cv.add(frame, mask)
    
    # Update previous frame
    prev_gray = gray.copy()
    prev_kp = next_kp.copy()
    
    cv.imshow('Tracked Points', output)
    if cv.waitKey(50) & 0xFF == ord('q'):
        break
    

cv.destroyAllWindows()