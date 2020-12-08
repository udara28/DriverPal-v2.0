"""
 MIT License

 Copyright (c) 2020 John T Vorhies

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
"""
import cv2 as cv
import numpy as np

# Set up dataset
num_frames = 438
path = "../../../../data/Test_Datasets/Kitti_2011_09_26_drive_0051/2011_09_26_drive_0051_sync/"
path_left = path + "rgb_left/data/"
path_right = path + "rgb_right/data/"

# Intrinsic parameters from calibration data
f = 721.5377 # pixels, from EvaluationUtils
camera_distance = 0.54 # meters, from EvaluationUtils. Camera 2 -> 3

for frame_iter in range(0,num_frames):

    frame_iter_str = str(frame_iter).zfill(10)
    frame_l = cv.imread(path_left + frame_iter_str +".png")
    gray_l = cv.cvtColor(frame_l, cv.COLOR_BGR2GRAY)
    frame_r = cv.imread(path_right + frame_iter_str +".png")
    gray_r = cv.cvtColor(frame_r, cv.COLOR_BGR2GRAY)

    stereo = cv.StereoBM_create(numDisparities=96, blockSize=21)
    disparity = stereo.compute(gray_l,gray_r)
    # Output is [16,4] fixed-point. Convert to float
    disparity = np.array(disparity)
    disparity = disparity.astype(np.float32)/16 + 1

    # Calculating depth might be something like
    # For each bounding box
    #   Compute average disparity
    #   Depth = f*camera_distance/average disparity
    #   Convert from pixels to meters?

    depth_map = (disparity - stereo.getMinDisparity())/stereo.getNumDisparities()
    cv.imshow('Disparity Map', depth_map)
    if cv.waitKey(50) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
