#include <iostream>
#include <string>

#include <opencv2/calib3d.hpp> // SGBM
#include <opencv2/imgcodecs.hpp> // imread
#include <opencv2/imgproc.hpp> // rgb2gray
#include <opencv2/highgui.hpp> // namedWindow

int main()
{
    // Set up dataset
    int num_frames = 438;
    std::string path = "/home/john/Documents/Xilinx_Contest/DriverPal-v2.0/estimateDistVel/Test_Datasets/Kitti_2011_09_26_drive_0051/2011_09_26_drive_0051_sync/";
    std::string path_left = path + "rgb_left/data/";
    std::string path_right = path + "rgb_right/data/";
    std::string img_ext = ".png";

    // Stereo matching parameters
    int block_size = 9;
    int num_disp = 96;
    auto stereo = cv::StereoBM::create(num_disp,block_size);

    for (int frame_iter = 0; frame_iter < num_frames; frame_iter++) {
        // Read in left and right frames
        std::string frame_iter_str = std::to_string(frame_iter);
        frame_iter_str =  std::string(10 - frame_iter_str.length(), '0') + frame_iter_str;
        cv::Mat frame_l = cv::imread((path_left+frame_iter_str+img_ext));
        cv::Mat frame_r = cv::imread((path_right+frame_iter_str+img_ext));

        if ( !frame_l.data || !frame_r.data ) { 
            std::cout << " --(!) Error reading images " << std::endl; 
            return -1;
        }

        cv::Mat gray_l, gray_r;
        cv::cvtColor(frame_l, gray_l, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame_r, gray_r, cv::COLOR_BGR2GRAY);

        // Compute disparity
        cv::Mat disp_map, depth_map;
        stereo->compute(gray_l,gray_r,disp_map);
        disp_map.convertTo(depth_map, CV_32F);
        depth_map = depth_map/(num_disp*16.0);

        // Display left frame
        cv::namedWindow("Video Feed",cv::WINDOW_AUTOSIZE);
        cv::imshow("Video Feed",depth_map);
    
        int key_press = cv::waitKey(1);
        if (key_press == 'q') {
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}