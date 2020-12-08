/*
 *
 * MIT License
 * 
 * Copyright (c) 2020 Udara De Silva (https://udaradesilva.com)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <cstring>
#include <iostream>
#include <stdio.h>
#include <map>
#include <math.h>
#include <type_traits>

#include <vitis/ai/nnpp/yolov3.hpp>
#include <vitis/ai/yolov3.hpp>

using namespace std;
using namespace cv;

// Structures for Bounding Box
struct BoundingBox {
    int xmin;
    int ymin;
    int xmax;
    int ymax;
    int label;
};
//----------------------

// Model Parameters
const double meters_per_pixel = 0.05;

//----------------------


// Function declerations
void second_stage(Mat&, vector<BoundingBox>&);
Mat optional_lane_detection(Mat&);
// ---------------------


Point onLine(Point p1, Point p2, int ycoord){
    Point p3(p1.x + int((p2.x-p1.x)/float(p2.y-p1.y+0.0000001)*(ycoord-p1.y)), ycoord);
    return p3;
}

int main(int argc, char** argv){
    cout << "Software written by Udara De Silva (https://udaradesilva.com)" << endl;
    cout << "\t -Xilinx Adaptive Computing Challenge 2020" << endl;

    // YOLOv3 Model
    auto yolo = vitis::ai::YOLOv3::create("yolov3_adas_pruned_0_9", true);
    vector<Point> bbTopLeft;
    vector<Point> bbBottomRight;

    Mat frame;

    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap("deep_dive_conv2.avi");
    
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    //--- GRAB AND WRITE LOOP
    cap.read(frame);
    cout << "Start grabbing" << endl
        << "Press any key to terminate" << endl
        << "Size height " << frame.rows << "\t width " << frame.cols << endl; 
    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);

        // check if we succeeded
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            //continue;
            break;
        }
        
        /*
         * YOLO v3 code
         */
        auto results = yolo->run(frame);
        
        vector<BoundingBox> bbArray;

        for(auto &box : results.bboxes){
            int label = box.label;
            float xmin = box.x * frame.cols + 1;
            float ymin = box.y * frame.rows + 1;
            float xmax = xmin + box.width * frame.cols;
            float ymax = ymin + box.height * frame.rows;
            if(xmin < 0.) xmin = 1.;
            if(ymin < 0.) ymin = 1.;
            if(xmax > frame.cols) xmax = frame.cols;
            if(ymax > frame.rows) ymax = frame.rows;
            float confidence = box.score;

            cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t"
                << xmax << "\t" << ymax << "\t" << confidence << "\n";

            BoundingBox b;
            b.xmin = (int)xmin;
            b.ymin = (int)ymin;
            b.xmax = (int)xmax;
            b.ymax = (int)ymax;
            b.label = label;
            bbArray.push_back(b);

        }       

        second_stage(frame, bbArray);

        if (waitKey(5) >= 0)
            break;
    }
}

void second_stage(Mat& frame, vector<BoundingBox>& bbArray){
    
    Mat lineFrame;

    // Lane lines
    vector<Vec4i> linesP;

    lineFrame = frame;
    //lineFrame = optional_lane_detection(frame);

    // Src trapezoidal reigon
    //-----------------------

    Point vanPoint = Point(630, 420);
    int width = 1280;
    int height = 720;
    Point pt1 = Point(vanPoint.x - width/4, vanPoint.y + 30);
    Point pt2 = Point(vanPoint.x + width/4, vanPoint.y + 30);
    Point pt3 = onLine(pt2, vanPoint, height - 35);
    Point pt4 = onLine(pt1, vanPoint, height - 35);


    line( lineFrame, pt1, pt2, Scalar(255,0,0), 1, LINE_AA);
    line( lineFrame, pt2, pt3, Scalar(255,0,0), 1, LINE_AA);
    line( lineFrame, pt3, pt4, Scalar(255,0,0), 1, LINE_AA);
    line( lineFrame, pt4, pt1, Scalar(255,0,0), 1, LINE_AA);

    // ----------------------

    // Dst trapezoidal reigon
    //-----------------------
    Point pd1 = Point(0, 0);
    Point pd2 = Point(600, 0);
    Point pd3 = Point(600, 720);
    Point pd4 = Point(0, 720);
    // ----------------------

    // Perspective transform
    // ---------------------
    vector<Point2f> src;
    vector<Point2f> dst;

    src.push_back(pt1);
    src.push_back(pt2);
    src.push_back(pt3);
    src.push_back(pt4);

    dst.push_back(pd1);
    dst.push_back(pd2);
    dst.push_back(pd3);
    dst.push_back(pd4);

    Mat lambda;
    Mat birdEyeView;
    lambda = getPerspectiveTransform(src, dst);
    warpPerspective(lineFrame, birdEyeView, lambda, Size(600, 720));

    // Bounding Box addition
    for(auto box : bbArray){
        if (box.label == 0) {
            rectangle(lineFrame, Point(box.xmin, box.ymin), Point(box.xmax, box.ymax), Scalar(0, 255, 0),
                    1, 1, 0);
        
            Point boxMidPoint(box.xmin + (box.xmax-box.xmin)/2  , (int)(box.ymax - 0.2*(box.ymax-box.ymin)));

            // Only considers objects in the trapezoidal reigon for the distance calculation
            if(boxMidPoint.y > (vanPoint.y + 30)){
                //warp object map
                float ka[3][1] = {{(float)boxMidPoint.x}, {(float)boxMidPoint.y}, {1}};
                Mat lambdaFloat;
                lambda.convertTo(lambdaFloat, CV_32FC1);
                Mat k = Mat(3, 1, CV_32FC1, ka);
                Mat kb = lambdaFloat * k; 
                Point objPoint(kb.at<float>(0,0)/(kb.at<float>(2,0)+0.0000001), kb.at<float>(1,0)/(kb.at<float>(2,0)+0.0000001));

                circle(birdEyeView, objPoint, 4, Scalar(255, 255, 255), FILLED, LINE_8, 0);
                line(birdEyeView, objPoint, Point(600/2, 720), Scalar(255,255, 255), 1, LINE_AA );

                //calculate distance
                double dist = sqrt(pow(objPoint.x - 600/2, 2) + pow(objPoint.y - 720, 2));
                dist *= meters_per_pixel;
                cout << "Distance to the box " << dist << endl;
                char distStr[50];
                sprintf(distStr, "%.2f m", dist);
                Point lblPoint(box.xmin+10, box.ymax-20);
                putText(lineFrame, distStr, lblPoint, FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,255,255), 2);
                // ---------------------

            }

        } else if (box.label == 1) {
            rectangle(lineFrame, Point(box.xmin, box.ymin), Point(box.xmax, box.ymax), Scalar(255, 0, 0),
                    1, 1, 0);
        } else if (box.label == 2) {
            rectangle(lineFrame, Point(box.xmin, box.ymin), Point(box.xmax, box.ymax), Scalar(0, 0, 255),
                    1, 1, 0);
        } else if (box.label == 3) {
            rectangle(lineFrame, Point(box.xmin, box.ymin), Point(box.xmax, box.ymax),
                    Scalar(0, 255, 255), 1, 1, 0);
        }

    }



    // Combine two views
    //----------------------
    Mat matDst(Size(1280+600,720),lineFrame.type(),Scalar::all(0));

    Mat leftFig = matDst(Rect(0,0, 1280, 720));
    lineFrame.copyTo(leftFig);

    Mat rightFig = matDst(Rect(1280, 0, 600, 720));
    birdEyeView.copyTo(rightFig);

    //----------------------

    imshow("Live", matDst);

}

Mat optional_lane_detection(Mat& frame) {


    Mat hlsFrame, edgeFrame, roi, roiEdges, lineFrame;

    // ROI points : (1280 x 720)
    Point roi_points[1][3];
    roi_points[0][0] = Point(1280/2, 720/2+50 );
    roi_points[0][1] = Point(0, 690);
    roi_points[0][2] = Point(1279, 690);
    const Point* ppt[1] = { roi_points[0] };
    int npt[] = { 3 };

    cvtColor(frame, hlsFrame, COLOR_RGB2HLS);
    Canny(hlsFrame, edgeFrame, 200, 100);
    roi = Mat::zeros(Size(frame.cols, frame.rows), CV_8U);
    fillPoly(roi, ppt, npt, 1, Scalar(255, 255, 255), LINE_8);

    roiEdges = edgeFrame.mul(roi);

    // Lane detection

    // Probabilistic Line Transform
    lineFrame = frame.clone();
    vector<Vec4i> linesP; // will hold the results of the detection
    //HoughLinesP(roiEdges, linesP, 1, CV_PI/180, 50, 50, 10 ); // runs the actual detection
    HoughLinesP(roiEdges, linesP, 1, CV_PI/180, 50, 180, 120 ); // runs the actual detection

    // Calculate vanishing point
    int Lhs = 0;
    Mat Rhs(2, 1, CV_32FC1, {{0}, {0}});
    // Draw the lines
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        Vec4i l = linesP[i];
        Point p1, p2;
        p1 = Point(l[0], l[1]);
        p2 = Point(l[2], l[3]);
        float angle = atan2(p1.y - p2.y, p1.x - p2.x);
        //cout << "Angle of line " << angle << endl;
        if(angle < 2.5 && angle > -2.5){
            float a[2][1] = {{(float) l[0]+100}, {(float) l[3]+180}};
            float b[2][1] = {{(float) l[2]+100}, {(float) l[1]+180}};
            Mat pn1 = Mat(2, 1, CV_32FC1, a);
            Mat pn2 = Mat(2, 1, CV_32FC1, b);
            Mat normV;
            normV = pn2 - pn1;    // vector normal to the line
            normV = normV / norm(normV + 0.00000001);
            int tmp = int(Mat(normV.t() * normV).at<float>(0,0));
            Lhs += tmp;
            Rhs += tmp * pn1;
            line( lineFrame, p1, p2, Scalar(0,0,255), 2, LINE_AA);
            /*
               if(angle > 0)
               line( lineFrame, Point(1280/2, 690), Point(1280/2 - normV[0], 690 - normV[1]), Scalar(0,255,0), 3, LINE_AA);
               else
               line( lineFrame, Point(1280/2, 690), Point(1280/2 + normV[0], 690 + normV[1]), Scalar(0,255,0), 3, LINE_AA);
               */
        }
    }
    // --------------

    // Vanishing point
    Mat vanMat = (1/(Lhs+0.00000001)) * Rhs;
    //circle(lineFrame, Point(int(vanMat.at<float>(1, 0)), int(vanMat.at<float>(0,0))), 20, Scalar(0,255, 0), FILLED, LINE_8, 0);
    circle(lineFrame, Point(630, 420), 4, Scalar(255,0, 0), FILLED, LINE_8, 0);
    //cout << "Vanishing point is at " << int(vanMat.at<float>(0,0)) << "," << int(vanMat.at<float>(1,0)) << endl;

    return lineFrame;

}
