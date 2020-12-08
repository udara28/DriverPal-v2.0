g++ main_app.cpp -g -I /usr/local/include/opencv4/ -L /usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_video

#g++ opencv_stat.cpp -I /usr/local/include/opencv4/ -L /usr/local/lib -lopencv_core -o opencv_stats


# This is the video converter command
# gst-launch-1.0 filesrc location=deep_dive_1.mov ! decodebin ! videoflip method=counterclockwise ! videoconvert ! jpegenc ! avimux ! filesink location=deep_dive_conv2.avi
