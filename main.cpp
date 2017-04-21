#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include <iostream>

using namespace cv;
using namespace std;




static void doIteration(Mat & img2, Ptr<FeatureDetector>& detector)
{

    vector<KeyPoint> keypoints;
    detector->detect( img2, keypoints );

    Mat original = img2.clone();
    Mat morph = img2.clone();
    Mat rope = img2.clone();
    //Grayscale matrix
    cv::Mat grayscaleMat (img2.size(), CV_8U);
    cv::cvtColor( img2, grayscaleMat, CV_BGR2GRAY);
    cv::Mat binaryMat(grayscaleMat.size(), CV_8U);
    cv::adaptiveThreshold(grayscaleMat, img2 ,100, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,21,-5);


    vector<KeyPoint> keypoints3;
    vector<KeyPoint> ropeKeypoints;
    uchar black = 0;
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

    //imshow("threshhold2", rope);
    erode( rope, rope, erosionElement );
    dilate( rope, rope, delutionElement );

    //imshow("threshhold2", rope);
    cv::Mat grayscaleMatRope2 (rope.size(), CV_8U);
    cv::cvtColor( rope, grayscaleMatRope2, CV_BGR2GRAY);
    cv::Mat binaryMatRope2(grayscaleMatRope2.size(), CV_8U);
    //imshow("threshhold3", grayscaleMatRope2);
    cv::adaptiveThreshold(grayscaleMatRope2, rope ,207, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,29,-3);


    erode( rope, rope, erosionElement2 );
    dilate( rope, rope, delutionElement );
    
    cv::cvtColor(rope,rope,CV_GRAY2BGR);
    // match keypoint to threshhold image 
    for (size_t i = 0; i < keypoints.size(); ++i){
         Point2f pixel = keypoints[i].pt;
         Vec3b colour =  img2.at<Vec3b>(pixel.y,pixel.x);
         Vec3b ropeColor = rope.at<Vec3b>(pixel.y,pixel.x);
         cout << ropeColor << endl;
         if(colour.val[0] != black){
                keypoints3.push_back(keypoints[i]);
                //circle(keypointsImg2, pixel, 1, CV_RGB(255,156,0),2);
            }   

        if(ropeColor.val[0] != black ){
            ropeKeypoints.push_back(keypoints[i]);
            circle(original, pixel, 1, CV_RGB(0,0,255),2);
        }
         
    }




    drawKeypoints(original, keypoints3,original, CV_RGB(255,156,0));
    //drawKeypoints(rope, ropeKeypoints,rope, CV_RGB(255,0,0));
    imshow( "winName2", original);
    //imshow( "winName2", rope);

    
}



int main(int argc, char** argv)
{

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
