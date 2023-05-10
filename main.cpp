#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;
int main()
{
    VideoCapture capture("video.mp4");
    if (!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open file!" << endl;
        return 0;
    }

    Mat img = imread("frame1.jpg");


    Mat frame1, oldFrame;
    capture >> frame1;
    auto roi = selectROI("select", frame1);
    cvtColor(frame1, oldFrame, COLOR_BGR2GRAY);
    while(true){
        Mat newFrame, newGray;
        capture >> newFrame;
        if (newFrame.empty())
            break;
        cvtColor(newFrame, newGray, COLOR_BGR2GRAY);
        Mat flow(oldFrame.size(), CV_32FC2);
        calcOpticalFlowFarneback(oldFrame, newGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        Mat flow_parts[2];
        split(flow, flow_parts);

        int x = roi.x;
        int y = roi.y;

        Mat xSubMat (flow_parts[0], roi);
        Mat ySubMat(flow_parts[1], roi);
        double dx = sum(xSubMat)[0]/(x*y);
        double dy = sum(ySubMat)[0]/(x*y);

        roi.x+=dx;
        roi.y+=dy;

        rectangle(newFrame, roi, cv::Scalar(0, 255, 0));

        imshow("newFrame", newFrame);

        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
        oldFrame = newGray;
    }
}