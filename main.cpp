#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help(char** argv)
{
    cout << "\nThis program demonstrats keypoint finding and matching between 2 images using features2d framework.\n"
     << "   In one case, the 2nd image is synthesized by homography from the first, in the second case, there are 2 images\n"
     << "\n"
     << "Case1: second image is obtained from the first (given) image using random generated homography matrix\n"
     << argv[0] << " [detectorType] [descriptorType] [matcherType] [matcherFilterType] [image] [evaluate(0 or 1)]\n"
     << "Example of case1:\n"
     << "./descriptor_extractor_matcher SURF SURF FlannBased NoneFilter cola.jpg 0\n"
     << "\n"
     << "Case2: both images are given. If ransacReprojThreshold>=0 then homography matrix are calculated\n"
     << argv[0] << " [detectorType] [descriptorType] [matcherType] [matcherFilterType] [image1] [image2] [ransacReprojThreshold]\n"
     << "\n"
     << "Matches are filtered using homography matrix in case1 and case2 (if ransacReprojThreshold>=0)\n"
     << "Example of case2:\n"
     << "./descriptor_extractor_matcher SURF SURF BruteForce CrossCheckFilter cola1.jpg cola2.jpg 3\n"
     << "\n"
     << "Possible detectorType values: see in documentation on createFeatureDetector().\n"
     << "Possible descriptorType values: see in documentation on createDescriptorExtractor().\n"
     << "Possible matcherType values: see in documentation on createDescriptorMatcher().\n"
     << "Possible matcherFilterType values: NoneFilter, CrossCheckFilter." << endl;
}



const string winName2 = "single";

static void doIteration(Mat& img2, Ptr<FeatureDetector>& detector)
{

    vector<KeyPoint> keypoints;
    detector->detect( img2, keypoints );

    Mat keypointsImg2 = img2;
    
    //Grayscale matrix
    cv::Mat grayscaleMat (img2.size(), CV_8U);
    cv::cvtColor( img2, grayscaleMat, CV_BGR2GRAY);
    cv::Mat binaryMat(grayscaleMat.size(), CV_8U);
    cv::adaptiveThreshold(grayscaleMat, img2 ,100, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,21,-5);


    vector<KeyPoint> keypoints3;
    uchar black = 0;
    cv::cvtColor(img2,img2,CV_GRAY2BGR);


    for (size_t i = 0; i < keypoints.size(); ++i){
         Point2f pixel = keypoints[i].pt;
         Vec3b colour =  img2.at<Vec3b>(pixel.y,pixel.x);
         
         if(colour.val[0] != black){
            circle(keypointsImg2, pixel, 1, CV_RGB(255,156,0),2);
            keypoints3.push_back(keypoints[i]);
         }
    }
    imshow( winName2, keypointsImg2);
}



int main(int argc, char** argv)
{
    if( argc != 7 && argc != 8 )
    {
        help(argv);
        return -1;
    }

    cv::initModule_nonfree();

    bool isWarpPerspective = argc == 7;
    double ransacReprojThreshold = -1;
    if( !isWarpPerspective )
        ransacReprojThreshold = atof(argv[7]);

    cout << "< Creating detector, descriptor extractor and descriptor matcher ..." << endl;
    Ptr<FeatureDetector> detector = FeatureDetector::create( argv[1] );


    cout << "< Reading the images..." << endl;
    Mat img1 = imread( argv[5] ), img2;
    if( !isWarpPerspective )
        img2 = imread( argv[6] );




    namedWindow(winName2, 1);

    doIteration(img2,detector);
    for(;;)
    {
        char c = (char)waitKey(0);
        if( c == '\x1b' ) // esc
        {
            cout << "Exiting ..." << endl;
            break;
        }
        else if( isWarpPerspective )
        {
            doIteration( img2,detector);
        }
    }
    return 0;
}
