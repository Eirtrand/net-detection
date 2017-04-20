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

char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";

    int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;
Mat src, src_gray, dst;
char* window_name = "Threshold Demo";


enum { NONE_FILTER = 0, CROSS_CHECK_FILTER = 1 };

static int getMatcherFilterType( const string& str )
{
    if( str == "NoneFilter" )
        return NONE_FILTER;
    if( str == "CrossCheckFilter" )
        return CROSS_CHECK_FILTER;
    CV_Error(CV_StsBadArg, "Invalid filter name");
    return -1;
}

static void simpleMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                     const Mat& descriptors1, const Mat& descriptors2,
                     vector<DMatch>& matches12 )
{
    vector<DMatch> matches;
    descriptorMatcher->match( descriptors1, descriptors2, matches12 );
}

static void crossCheckMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                         const Mat& descriptors1, const Mat& descriptors2,
                         vector<DMatch>& filteredMatches12, int knn=1 )
{
    filteredMatches12.clear();
    vector<vector<DMatch> > matches12, matches21;
    descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
    descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );
    for( size_t m = 0; m < matches12.size(); m++ )
    {
        bool findCrossCheck = false;
        for( size_t fk = 0; fk < matches12[m].size(); fk++ )
        {
            DMatch forward = matches12[m][fk];

            for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
            {
                DMatch backward = matches21[forward.trainIdx][bk];
                if( backward.trainIdx == forward.queryIdx )
                {
                    filteredMatches12.push_back(forward);
                    findCrossCheck = true;
                    break;
                }
            }
            if( findCrossCheck ) break;
        }
    }
}




static void doIteration( const Mat& img1, Mat& img2, bool isWarpPerspective,
                  vector<KeyPoint>& keypoints1, const Mat& descriptors1,
                  Ptr<FeatureDetector>& detector, Ptr<DescriptorExtractor>& descriptorExtractor,
                  Ptr<DescriptorMatcher>& descriptorMatcher, int matcherFilter, bool eval,
                  double ransacReprojThreshold, RNG& rng )
{

    vector<KeyPoint> keypoints;
    detector->detect( img2, keypoints );

    Mat keypointsImg2 = img2;
    //Grayscale matrix
    cv::Mat grayscaleMat (img2.size(), CV_8U);

    //Convert BGR to Gray
    cv::cvtColor( img2, grayscaleMat, CV_BGR2GRAY);

    //Binary image
    cv::Mat binaryMat(grayscaleMat.size(), CV_8U);

    //Apply thresholding
    cv::adaptiveThreshold(grayscaleMat, img2 ,100, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,21,-5);

    //imshow("original", grayscaleMat);
    
    //vector<Point2f> points2; KeyPoint::convert(keypoints, points2, trainIdxs);


    //drawKeypoints(img2, keypoints,keypointsImg2, CV_RGB(255,0,0));
    //imshow("original", keypointsImg2);

    vector<KeyPoint> keypoints3;
    uchar black = 0;
    cv::cvtColor(img2,img2,CV_GRAY2BGR);

    //imshow(winName2, img2);

    for (size_t i = 0; i < keypoints.size(); ++i){
         Point2f pixel = keypoints[i].pt;
         Vec3b colour =  img2.at<Vec3b>(pixel.y,pixel.x);
         
         if(colour.val[0] != black){
            circle(keypointsImg2, pixel, 1, CV_RGB(255,156,0),2);
            keypoints3.push_back(keypoints[i]);
            //cout << "Pixel: " << pixel << " with colour: " << colour << "    [DELETED]" << endl;
            
            //putText(img2, "white", pixel, FONT_HERSHEY_PLAIN, 0.2, CV_RGB(0,255,0), 1.0);
         }
         else{
            //cout << "Pixel: " << pixel << " with colour " << colour << endl;

         }     
    }




    //Mat keypointsImg;
    //drawKeypoints(binaryMat, keypoints3,keypointsImg, CV_RGB(255,0,0));
    //drawKeypoints(keypointsImg, keypoints3,keypointsImg, CV_RGB(0,255,0));
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
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create( argv[2] );
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( argv[3] );
    int mactherFilterType = getMatcherFilterType( argv[4] );
    bool eval = !isWarpPerspective ? false : (atoi(argv[6]) == 0 ? false : true);
    cout << ">" << endl;
    if( detector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty()  )
    {
        cout << "Can not create detector or descriptor exstractor or descriptor matcher of given types" << endl;
        return -1;
    }

    cout << "< Reading the images..." << endl;
    Mat img1 = imread( argv[5] ), img2;
    if( !isWarpPerspective )
        img2 = imread( argv[6] );
    cout << ">" << endl;
    if( img1.empty() || (!isWarpPerspective && img2.empty()) )
    {
        cout << "Can not read images" << endl;
        return -1;
    }

    cout << endl << "< Extracting keypoints from first image..." << endl;
    vector<KeyPoint> keypoints1;
    detector->detect( img1, keypoints1 );
    cout << keypoints1.size() << " points" << endl << ">" << endl;

    cout << "< Computing descriptors for keypoints from first image..." << endl;
    Mat descriptors1;
    descriptorExtractor->compute( img1, keypoints1, descriptors1 );
    cout << ">" << endl;

    namedWindow(winName2, 1);
    RNG rng = theRNG();
    doIteration( img1, img2, isWarpPerspective, keypoints1, descriptors1,
                 detector, descriptorExtractor, descriptorMatcher, mactherFilterType, eval,
                 ransacReprojThreshold, rng );
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
            doIteration( img1, img2, isWarpPerspective, keypoints1, descriptors1,
                         detector, descriptorExtractor, descriptorMatcher, mactherFilterType, eval,
                         ransacReprojThreshold, rng );
        }
    }
    return 0;
}

