#include <opencv2/highgui/highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <chrono>
#include <ctime>

#define USE249
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	std::cout << "OpenCV version: "
		<< CV_MAJOR_VERSION << "."
		<< CV_MINOR_VERSION << "."
		<< CV_SUBMINOR_VERSION
		<< std::endl;
	cv::Mat im = cv::imread("adam1.png", 1);
	if (im.empty())
	{
		cout << "Cannot open image!" << endl;
		return -1;
	}
	cv::Mat gray;
	cv::cvtColor(im, gray, cv::COLOR_BGR2GRAY);
	int mnArea = 40 * 40;
	int mxArea = im.rows*im.cols*0.4;
	std::vector< std::vector< cv::Point > > ptblobs;
	std::vector<cv::Rect> bboxes;
	std::chrono::time_point<std::chrono::system_clock> start, end;

	start = std::chrono::system_clock::now();
#ifndef USE249
	cv::Ptr<cv::MSER> mser = cv::MSER::create(1, mnArea, mxArea, 0.25, 0.2);
	mser->detectRegions(gray, ptblobs, bboxes);
#else
	
	cv::MserFeatureDetector mser(1, mnArea, mxArea, 0.25, 0.2);
	mser(gray, ptblobs); 
#endif
	end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);

	std::cout << "finished computation at " << std::ctime(&end_time)
		<< "elapsed time: " << elapsed_seconds.count() << "s\n";

	cv::namedWindow("image", cv::WINDOW_NORMAL);
	cv::imshow("image", im);
	cv::waitKey(0);

	return 0;
}