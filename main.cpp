#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <algorithm>    // std::sort
#include <bits/stdc++.h>
#include <string>
#include <sstream>


#include <iostream>

using namespace cv;
using namespace std;

bool response_comparator(const KeyPoint& p1, const KeyPoint& p2) {
    return p1.pt.x > p2.pt.x;
}



static void doIteration(Mat & img2, Ptr<FeatureDetector>& detector, cv::VideoWriter & output_cap)
{

    vector<KeyPoint> keypoints;
    vector<KeyPoint> keypoints3;
    vector<KeyPoint> ropeKeypoints;


    detector->detect( img2, keypoints );

    Mat original = img2.clone();
    Mat morph = img2.clone();
    Mat rope = img2.clone();

    //Grayscale matrix
    cv::Mat grayscaleMat (img2.size(), CV_8U);
    cv::cvtColor( img2, grayscaleMat, CV_BGR2GRAY);
    cv::Mat binaryMat(grayscaleMat.size(), CV_8U);
    cv::adaptiveThreshold(grayscaleMat, img2 ,100, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,21,-5);


    cv::cvtColor(img2,img2,CV_GRAY2BGR);


    int erosion_elem = 1;
    int erosion_size = 8;
    int erosion_size2 = 4;
    int dilation_elem = 2;
    int dilation_size = 12;


    int morph_elem = 0;
    int morph_size = 10;
    int morph_operator = 0;

    

    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    /// Apply the specified morphology operation

    morphologyEx( morph, rope, morph_operator + 2, element );
    

    Mat delutionElement = getStructuringElement(dilation_elem,Size( 2*dilation_size + 1, 2*dilation_size+1 ), Point( dilation_size, dilation_size )); 
    Mat erosionElement = getStructuringElement(erosion_elem,Size( 2*erosion_size + 1, 2*erosion_size+1 ), Point( erosion_size, erosion_size ) );
    Mat erosionElement2 = getStructuringElement(2,Size( 2*erosion_size2 + 1, 2*erosion_size2+1 ), Point( erosion_size2, erosion_size2 ) );


    erode( rope, rope, erosionElement );
    dilate( rope, rope, delutionElement );


    cv::Mat grayscaleMatRope2 (rope.size(), CV_8U);
    cv::cvtColor( rope, grayscaleMatRope2, CV_BGR2GRAY);
    cv::Mat binaryMatRope2(grayscaleMatRope2.size(), CV_8U);
    cv::adaptiveThreshold(grayscaleMatRope2, rope ,207, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,29,-3);

    erode( rope, rope, erosionElement2 );
    dilate( rope, rope, delutionElement );
    
    cv::cvtColor(rope,rope,CV_GRAY2BGR);


    // match keypoint to threshhold image 
    uchar black = 0;
    keypoints.erase(keypoints.end() - keypoints.size()/5, keypoints.end() - 1);

    for (size_t i = 0; i < keypoints.size(); ++i){
         Point2f pixel = keypoints[i].pt;
         Vec3b colour =  img2.at<Vec3b>(pixel.y,pixel.x);
         Vec3b ropeColor = rope.at<Vec3b>(pixel.y,pixel.x);

        if(ropeColor.val[0] != black ){
            ropeKeypoints.push_back(keypoints[i]);
            circle(original, pixel, 3, CV_RGB(0,0,255),2);
        }
         else if(colour.val[0] != black){
                keypoints3.push_back(keypoints[i]);
                //circle(keypointsImg2, pixel, 1, CV_RGB(255,156,0),2);
                //cout << keypoints[i].response << endl;
            }   


         
    }


    //calculate distance
   std::sort(keypoints3.begin(), keypoints3.end(), response_comparator);
   float distance [keypoints3.size()];

   for (size_t i = 1; i < keypoints3.size(); ++i){

        distance[i] = keypoints3[i].pt.x-keypoints3[i-1].pt.x;
        //cout << distance[i] << endl;
   }

   int n = sizeof(distance)/sizeof(distance[0]);
   sort(distance, distance+n); 
   cout << "Distance:  " << -0.3/distance[int(n/2)] << endl;

    std::stringstream ss;
    ss << "Distance to net: " << -0.3/distance[int(n/2)] << "m";
    string disp = ss.str();

    putText(original, disp, Point(50, 100), FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255,0,0), 4);
    drawKeypoints(original, keypoints3,original, CV_RGB(255,156,0));
    //drawKeypoints(rope, ropeKeypoints,rope, CV_RGB(255,0,0));
    output_cap.write(original);
    //imshow( "winName2", original);
    //imshow( "winName2", rope);

    
}



int main(int argc, char** argv)
{


    cv::initModule_nonfree();

    bool isWarpPerspective = argc == 7;
    double ransacReprojThreshold = -1;
    if( !isWarpPerspective )
        ransacReprojThreshold = atof(argv[7]);

    Ptr<FeatureDetector> detector = FeatureDetector::create( argv[1] );

    Mat img2;
    if( !isWarpPerspective )
        img2 = imread( argv[6] );


cout << "opening video" << endl;
 VideoCapture inputVideo( "../night.mp4");
    if(!inputVideo.isOpened()) { // check if we succeeded
        cout << "fail open video" << endl;
        return -1;
    }

cv::VideoWriter output_cap("../video.avi", CV_FOURCC('M','J','P','G'), 30, Size(1920, 1080));
int i = 0;
    for(;;){

        Mat frame;
        inputVideo >> frame; 
        if(i == 800) break;
        doIteration(frame, detector, output_cap );
        if(waitKey(1) >= 0) break;
        i++;
    }

    output_cap.release();
return 0;
}
