#pragma once

#define USE_DARKNET 1
#define HAVE_STRUCT_TIMESPEC 1

#define GPU 1 
#define CUDNN 1

#ifdef USE_DARKNET
#include "..\\TEST_DEMO\\my_algorithm\\darknet\\include\\darknet.h"
	#ifdef _DEBUG
	#pragma comment(lib,"..\\TEST_DEMO\\my_algorithm\\darknet\\lib\\darkd.lib")
	#else
	#pragma comment(lib,"..\\TEST_DEMO\\my_algorithm\\darknet\\lib\\dark.lib")
	#endif
#endif


#include "opencv2\\opencv.hpp"
#ifdef _DEBUG
#pragma comment(lib,"opencv_world349d.lib")
#else
#pragma comment(lib,"opencv_world349.lib")
#endif

#include <future>
#include "MY_QUEUE.h"

typedef struct iou_box {
	float left_x, right_x, top_y, bot_y;
} iou_box;

enum {TP,FP,FN,TN}; //true positive, false positive...

class MY_ALGORITHM
{
private:
	MY_ALGORITHM(){}
	
	MY_DATA m_data; //받은 데이터
	
	network* m_net;
	char** m_names;

public:
	cv::Mat Detect_Image_File(char *image_file);
	void DetectVideo(char *video_file);
	bool init(int gpu_index, char* names_file, char* cfg_file, char* weights_file);
	void FreeAll();

	My_Data Detect_Image(cv::Mat f);
	std::vector<iou_box> Detect_Image_for_IOU(cv::Mat f);
	void DetectFolder(char* folder);

	void ReadBoxfile(char* boxfile1, char* boxfile2);
};