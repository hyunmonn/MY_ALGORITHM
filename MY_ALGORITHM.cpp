#include "pch.h"
#include "MY_ALGORITHM.h"
#include <iostream>
#include <fstream>


//제작함
void get_name_list(char* name_file, int& names_size,char** names)
{
	std::ifstream readfile;
	readfile.open(name_file);

	names_size = 0;
	if (readfile.is_open())
	{
		while (!readfile.eof())
		{
			char t[256] = { 0 };
			readfile.getline(t, 256);
			memcpy(names[names_size++],t,sizeof(char)*256);
		}
	}
	readfile.close();
}


//복붙함
void calloc_error()
{
	fprintf(stderr, "Calloc error\n");
	exit(EXIT_FAILURE);
}

void* xcalloc(size_t nmemb, size_t size) {
	void* ptr = calloc(nmemb, size);
	if (!ptr) {
		calloc_error();
	}
	return ptr;
}

// Creates array of detections with prob > thresh and fills best_class for them
detection_with_class* get_actual_detections(detection* dets, int dets_num, float thresh, int* selected_detections_num, char** names)
{
	int selected_num = 0;
	detection_with_class* result_arr = (detection_with_class*)xcalloc(dets_num, sizeof(detection_with_class));
	int i;
	for (i = 0; i < dets_num; ++i) {
		int best_class = -1;
		float best_class_prob = thresh;
		int j;
		for (j = 0; j < dets[i].classes; ++j) {
			int show = strncmp(names[j], "dont_show", 9);
			if (dets[i].prob[j] > best_class_prob && show) {
				best_class = j;
				best_class_prob = dets[i].prob[j];
			}
		}
		if (best_class >= 0) {
			result_arr[selected_num].det = dets[i];
			result_arr[selected_num].best_class = best_class;
			++selected_num;
		}
	}
	if (selected_detections_num)
		*selected_detections_num = selected_num;
	return result_arr;
}

// compare to sort detection** by bbox.x
int compare_by_lefts(const void* a_ptr, const void* b_ptr) {
	const detection_with_class* a = (detection_with_class*)a_ptr;
	const detection_with_class* b = (detection_with_class*)b_ptr;
	const float delta = (a->det.bbox.x - a->det.bbox.w / 2) - (b->det.bbox.x - b->det.bbox.w / 2);
	return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

// compare to sort detection** by best_class probability
int compare_by_probs(const void* a_ptr, const void* b_ptr) {
	const detection_with_class* a = (detection_with_class*)a_ptr;
	const detection_with_class* b = (detection_with_class*)b_ptr;
	float delta = a->det.prob[a->best_class] - b->det.prob[b->best_class];
	return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}


float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

float get_color(int c, int x, int max)
{
	float ratio = ((float)x / max) * 5;
	int i = floor(ratio);
	int j = ceil(ratio);
	ratio -= i;
	float r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
	//printf("%f\n", r);
	return r;
}

void draw_box_bw(image a, int x1, int y1, int x2, int y2, float brightness)
{
	//normalize_image(a);
	int i;
	if (x1 < 0) x1 = 0;
	if (x1 >= a.w) x1 = a.w - 1;
	if (x2 < 0) x2 = 0;
	if (x2 >= a.w) x2 = a.w - 1;

	if (y1 < 0) y1 = 0;
	if (y1 >= a.h) y1 = a.h - 1;
	if (y2 < 0) y2 = 0;
	if (y2 >= a.h) y2 = a.h - 1;

	for (i = x1; i <= x2; ++i) {
		a.data[i + y1 * a.w + 0 * a.w * a.h] = brightness;
		a.data[i + y2 * a.w + 0 * a.w * a.h] = brightness;
	}
	for (i = y1; i <= y2; ++i) {
		a.data[x1 + i * a.w + 0 * a.w * a.h] = brightness;
		a.data[x2 + i * a.w + 0 * a.w * a.h] = brightness;
	}
}

void draw_box_width_bw(image a, int x1, int y1, int x2, int y2, int w, float brightness)
{
	int i;
	for (i = 0; i < w; ++i) {
		float alternate_color = (w % 2) ? (brightness) : (1.0 - brightness);
		draw_box_bw(a, x1 + i, y1 + i, x2 - i, y2 - i, alternate_color);
	}
}

void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
	//normalize_image(a);
	int i;
	if (x1 < 0) x1 = 0;
	if (x1 >= a.w) x1 = a.w - 1;
	if (x2 < 0) x2 = 0;
	if (x2 >= a.w) x2 = a.w - 1;

	if (y1 < 0) y1 = 0;
	if (y1 >= a.h) y1 = a.h - 1;
	if (y2 < 0) y2 = 0;
	if (y2 >= a.h) y2 = a.h - 1;

	for (i = x1; i <= x2; ++i) {
		a.data[i + y1 * a.w + 0 * a.w * a.h] = r;
		a.data[i + y2 * a.w + 0 * a.w * a.h] = r;

		a.data[i + y1 * a.w + 1 * a.w * a.h] = g;
		a.data[i + y2 * a.w + 1 * a.w * a.h] = g;

		a.data[i + y1 * a.w + 2 * a.w * a.h] = b;
		a.data[i + y2 * a.w + 2 * a.w * a.h] = b;
	}
	for (i = y1; i <= y2; ++i) {
		a.data[x1 + i * a.w + 0 * a.w * a.h] = r;
		a.data[x2 + i * a.w + 0 * a.w * a.h] = r;

		a.data[x1 + i * a.w + 1 * a.w * a.h] = g;
		a.data[x2 + i * a.w + 1 * a.w * a.h] = g;

		a.data[x1 + i * a.w + 2 * a.w * a.h] = b;
		a.data[x2 + i * a.w + 2 * a.w * a.h] = b;
	}
}

void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
	int i;
	for (i = 0; i < w; ++i) {
		draw_box(a, x1 + i, y1 + i, x2 - i, y2 - i, r, g, b);
	}
}

static float get_pixel(image m, int x, int y, int c)
{
	assert(x < m.w&& y < m.h&& c < m.c);
	return m.data[c * m.h * m.w + y * m.w + x];
}

static void set_pixel(image m, int x, int y, int c, float val)
{
	if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
	assert(x < m.w&& y < m.h&& c < m.c);
	m.data[c * m.h * m.w + y * m.w + x] = val;
}

void draw_label(image a, int r, int c, image label, const float* rgb)
{
	int w = label.w;
	int h = label.h;
	if (r - h >= 0) r = r - h;

	int i, j, k;
	for (j = 0; j < h && j + r < a.h; ++j) {
		for (i = 0; i < w && i + c < a.w; ++i) {
			for (k = 0; k < label.c; ++k) {
				float val = get_pixel(label, i, j, k);
				set_pixel(a, i + c, j + r, k, rgb[k] * val);
			}
		}
	}
}

image make_empty_image(int w, int h, int c)
{
	image out;
	out.data = 0;
	out.h = h;
	out.w = w;
	out.c = c;
	return out;
}

image float_to_image(int w, int h, int c, float* data)
{
	image out = make_empty_image(w, h, c);
	out.data = data;
	return out;
}

image threshold_image(image im, float thresh)
{
	int i;
	image t = make_image(im.w, im.h, im.c);
	for (i = 0; i < im.w * im.h * im.c; ++i) {
		t.data[i] = im.data[i] > thresh ? 1 : 0;
	}
	return t;
}

void embed_image(image source, image dest, int dx, int dy)
{
	int x, y, k;
	for (k = 0; k < source.c; ++k) {
		for (y = 0; y < source.h; ++y) {
			for (x = 0; x < source.w; ++x) {
				float val = get_pixel(source, x, y, k);
				set_pixel(dest, dx + x, dy + y, k, val);
			}
		}
	}
}

//거의 복붙..
void draw_detections_v3_custom(image im, detection* dets, int num, float thresh, char** names, /*image** alphabet,*/ int classes, int ext_output, My_Data &my_data)
{
	static int frame_id = 0;
	frame_id++;

	int selected_detections_num;
	detection_with_class* selected_detections = get_actual_detections(dets, num, thresh, &selected_detections_num, names);
	
	// text output
	qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);
	int i;

	for (i = 0; i < selected_detections_num; ++i) {
		const int best_class = selected_detections[i].best_class;
		printf("%s: %.0f%%", names[best_class], selected_detections[i].det.prob[best_class] * 100);
		if (ext_output)
			printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
				round((selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2) * im.w),
				round((selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2) * im.h),
				round(selected_detections[i].det.bbox.w * im.w), round(selected_detections[i].det.bbox.h * im.h));
		else
			printf("\n");
		int j;
		for (j = 0; j < classes; ++j) {
			if (selected_detections[i].det.prob[j] > thresh && j != best_class) {
				printf("%s: %.0f%%", names[j], selected_detections[i].det.prob[j] * 100);

				if (ext_output)
					printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
						round((selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2) * im.w),
						round((selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2) * im.h),
						round(selected_detections[i].det.bbox.w * im.w), round(selected_detections[i].det.bbox.h * im.h));
				else
					printf("\n");
			}
		}
	}

	qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_probs);
	for (i = 0; i < selected_detections_num; ++i) {
		int width = im.h * .006;
		if (width < 1)
			width = 1;

		//printf("%d %s: %.0f%%\n", i, names[selected_detections[i].best_class], prob*100);
		int offset = selected_detections[i].best_class * 123457 % classes;
		float red = get_color(2, offset, classes);
		float green = get_color(1, offset, classes);
		float blue = get_color(0, offset, classes);
		float rgb[3];

		rgb[0] = red;
		rgb[1] = green;
		rgb[2] = blue;
		box b = selected_detections[i].det.bbox;

		int left = (b.x - b.w / 2.) * im.w;
		int right = (b.x + b.w / 2.) * im.w;
		int top = (b.y - b.h / 2.) * im.h;
		int bot = (b.y + b.h / 2.) * im.h;

		if (left < 0) left = 0;
		if (right > im.w - 1) right = im.w - 1;
		if (top < 0) top = 0;
		if (bot > im.h - 1) bot = im.h - 1;

		if (im.c == 1) {
			draw_box_width_bw(im, left, top, right, bot, width, 0.8);    // 1 channel Black-White
		}
		else {
			draw_box_width(im, left, top, right, bot, width, red, green, blue); // 3 channels RGB

			My_Box m_b;
			m_b.x = b.x;
			m_b.y = b.y;
			m_b.w = b.w;
			m_b.h = b.h;
			my_data.boxes.push_back(m_b);
		}
		if (/*alphabet*/0) {//------- 1해도 결과는 같음
			char labelstr[4096] = { 0 };
			strcat(labelstr, names[selected_detections[i].best_class]);
			int j;
			for (j = 0; j < classes; ++j) {
				if (selected_detections[i].det.prob[j] > thresh && j != selected_detections[i].best_class) {
					strcat(labelstr, ", ");
					strcat(labelstr, names[j]);
				}
			}
			//image label = get_label_v3(alphabet, labelstr, (im.h * .03));			
			//draw_label(im, top + width, left, /*label*/im, rgb);			
			//free_image(label);
		}
		if (selected_detections[i].det.mask) {//실행 x
			image mask = float_to_image(14, 14, 1, selected_detections[i].det.mask);
			image resized_mask = resize_image(mask, b.w * im.w, b.h * im.h);
			image tmask = threshold_image(resized_mask, .5);
			embed_image(tmask, im, left, top);
			free_image(mask);
			free_image(resized_mask);
			free_image(tmask);
		}
	}
	free(selected_detections);
}

void draw_detections_v3_custom(image im, detection* dets, int num, float thresh, char** names, /*image** alphabet,*/ int classes, int ext_output, std::vector<iou_box>& my_boxes)
{
	static int frame_id = 0;
	frame_id++;

	int selected_detections_num;
	detection_with_class* selected_detections = get_actual_detections(dets, num, thresh, &selected_detections_num, names);

	// text output
	qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);
	int i;

	for (i = 0; i < selected_detections_num; ++i) {
		const int best_class = selected_detections[i].best_class;
		printf("%s: %.0f%%", names[best_class], selected_detections[i].det.prob[best_class] * 100);
		if (ext_output)
			printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
				round((selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2) * im.w),
				round((selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2) * im.h),
				round(selected_detections[i].det.bbox.w * im.w), round(selected_detections[i].det.bbox.h * im.h));
		else
			printf("\n");
		int j;
		for (j = 0; j < classes; ++j) {
			if (selected_detections[i].det.prob[j] > thresh && j != best_class) {
				printf("%s: %.0f%%", names[j], selected_detections[i].det.prob[j] * 100);

				if (ext_output)
					printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
						round((selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2) * im.w),
						round((selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2) * im.h),
						round(selected_detections[i].det.bbox.w * im.w), round(selected_detections[i].det.bbox.h * im.h));
				else
					printf("\n");
			}
		}
	}

	qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_probs);
	for (i = 0; i < selected_detections_num; ++i) {
		int width = im.h * .006;
		if (width < 1)
			width = 1;

		//printf("%d %s: %.0f%%\n", i, names[selected_detections[i].best_class], prob*100);
		int offset = selected_detections[i].best_class * 123457 % classes;
		float red = get_color(2, offset, classes);
		float green = get_color(1, offset, classes);
		float blue = get_color(0, offset, classes);
		float rgb[3];

		rgb[0] = red;
		rgb[1] = green;
		rgb[2] = blue;
		box b = selected_detections[i].det.bbox;

		int left = (b.x - b.w / 2.) * im.w;
		int right = (b.x + b.w / 2.) * im.w;
		int top = (b.y - b.h / 2.) * im.h;
		int bot = (b.y + b.h / 2.) * im.h;

		if (left < 0) left = 0;
		if (right > im.w - 1) right = im.w - 1;
		if (top < 0) top = 0;
		if (bot > im.h - 1) bot = im.h - 1;

		//추가
		iou_box my_box;

		my_box.left_x = left;
		my_box.right_x = right;
		my_box.top_y = top;
		my_box.bot_y = bot;

		my_boxes.push_back(my_box);

		if (im.c == 1) {
			draw_box_width_bw(im, left, top, right, bot, width, 0.8);    // 1 channel Black-White
		}
		else {
			draw_box_width(im, left, top, right, bot, width, red, green, blue); // 3 channels RGB
		}
	}
	free(selected_detections);
}


image copy_image(image p)
{
	image copy = p;
	copy.data = (float*)xcalloc(p.h * p.w * p.c, sizeof(float));
	memcpy(copy.data, p.data, p.h * p.w * p.c * sizeof(float));
	return copy;
}

void constrain_image(image im)
{
	int i;
	for (i = 0; i < im.w * im.h * im.c; ++i) {
		if (im.data[i] < 0) im.data[i] = 0;
		if (im.data[i] > 1) im.data[i] = 1;
	}
}

cv::Mat image_to_mat(image img)
{
	int channels = img.c;
	int width = img.w;
	int height = img.h;
	cv::Mat mat = cv::Mat(height, width, CV_8UC(channels));
	int step = mat.step;

	for (int y = 0; y < img.h; ++y) {
		for (int x = 0; x < img.w; ++x) {
			for (int c = 0; c < img.c; ++c) {
				float val = img.data[c * img.h * img.w + y * img.w + x];
				mat.data[y * step + x * img.c + c] = (unsigned char)(val * 255);
			}
		}
	}
	return mat;
}

cv::Mat show_image(image p, const char* name)
{
	cv::Mat mat;
	try {
		image copy = copy_image(p);
		constrain_image(copy);

		mat = image_to_mat(copy);
		if (mat.channels() == 3) cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
		else if (mat.channels() == 4) cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);
		cv::namedWindow(name, cv::WINDOW_NORMAL);
		cv::imshow(name, mat);
		free_image(copy);
	}
	catch (...) {
		std::cerr << "OpenCV exception: show_image_cv \n";
	}	
	return mat;
}

image mat_to_image(cv::Mat mat)
{
	int w = mat.cols;
	int h = mat.rows;
	int c = mat.channels();
	image im = make_image(w, h, c);
	unsigned char* data = (unsigned char*)mat.data;
	int step = mat.step;
	for (int y = 0; y < h; ++y) {
		for (int k = 0; k < c; ++k) {
			for (int x = 0; x < w; ++x) {
				//uint8_t val = mat.ptr<uint8_t>(y)[c * x + k];
				//uint8_t val = mat.at<Vec3b>(y, x).val[k];
				//im.data[k*w*h + y*w + x] = val / 255.0f;

				im.data[k * w * h + y * w + x] = data[y * step + x * c + k] / 255.0f;
			}
		}
	}
	return im;
}

image load_image(char* filename, int w, int h, int c)
{
//#ifdef OPENCV
	//image out = load_image_stb(filename, c);
	cv::Mat mat;
	if(c==0)
		mat = cv::imread(filename, 0); // gray 0 color 1 //load_image_mat(filename, c);
	else
		mat = cv::imread(filename, 1);
	if (mat.empty()) {
		return make_image(10, 10, c);
	}
	image out = mat_to_image(mat);
	
//#else
//	image out = load_image_stb(filename, c);    // without OpenCV
//#endif  // OPENCV

	if ((h && w) && (h != out.h || w != out.w)) {
		image resized = resize_image(out, w, h);
		free_image(out);
		out = resized;
	}
	return out;
}

bool MY_ALGORITHM::init(int gpu_index, char* names_file, char* cfg_file, char* weights_file)
{
	if (gpu_index >= 0) {
		cuda_set_device(gpu_index);
	}

	try {
		//name
		int names_size = 0;
		char* name_file = names_file;

		m_net = load_network(
			cfg_file,
			weights_file,
			0);
		fuse_conv_batchnorm(*m_net);
		calculate_binary_weights(*m_net);

		m_names = (char**)malloc(sizeof(char*) * 80); //최대 name은 80개로 잡아놓음
		for (int i = 0; i < 80; i++) //name의 이름 길이는 256을 넘으면 안 됨
			m_names[i] = (char*)malloc(sizeof(char) * 256);

		get_name_list(name_file, names_size, m_names);

		if (m_net->layers[m_net->n - 1].classes != names_size) {
			printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file [cfgfile] \n",
				name_file, names_size, m_net->layers[m_net->n - 1].classes);
			if (m_net->layers[m_net->n - 1].classes > names_size) getchar();
		}
		srand(2222222);
	}
	catch (int expn)
	{
		std::cout << "[error] : " << expn << std::endl;
		return false;
	}
	return true;
}

void MY_ALGORITHM::FreeAll()
{
	//free m
	for (int i = 0; i < 80; i++)
		free(m_names[i]);
	free(m_names);

	free_network(*m_net);
}

cv::Mat MY_ALGORITHM::Detect_Image_File(char * image_file)
{
	My_Data my_data;
	cv::Mat returnMat; 

	try {		
		//thresh, hier_thresh
		float thresh = 0.25, hier_thresh = 0.5;
				
		//gray 0 color 1
		cv::Mat mat;
		if (m_net->c == 0)
			mat = cv::imread(image_file, 0); // gray 0 color 1
		else
			mat = cv::imread(image_file, 1);
		
		//load_image(image_file, 0, 0,net->c); //file_path, resize_widht, resize_height 
		
		CV_Assert(!mat.empty()); // empty일 경우 Error , 내부가 false일 경우
		
		image im = mat_to_image(mat);

		image sized = resize_image(im, m_net->w, m_net->h);

		layer l = m_net->layers[m_net->n - 1];


		float* X = sized.data;

		double time = get_time_point();
		network_predict(*m_net, X);
		printf("%s: Predicted in %lf milli-seconds.\n", image_file, ((double)get_time_point() - time) / 1000);

		int nboxes = 0;
		detection* dets = get_network_boxes(m_net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, /*letter_box*/0);
		float nms = .45;    // 0.4F
		if (nms) {
			if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
			else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
		}

		draw_detections_v3_custom(im, dets, nboxes, thresh, m_names, /*alphabet,*/ l.classes, /*ext_output*/0, my_data);
		
		//save_image(im, "predictions");
		//returnMat = show_image(im, "predictions");

		returnMat = image_to_mat(im);

		free_detections(dets, nboxes);
		free_image(im);
		free_image(sized);

		//cv::waitKey(0);		
	}
	catch (int expn)
	{
		std::cout << "[error] : " << expn << std::endl;
	}
	return returnMat;
}

My_Data MY_ALGORITHM::Detect_Image(cv::Mat f)
{
	My_Data my_data;
	try {
		//thresh, hier_thresh
		float thresh = 0.25, hier_thresh = 0.5;

		CV_Assert(!f.empty()); // empty일 경우 Error , 내부가 false일 경우

		image im = mat_to_image(f);

		image sized = resize_image(im, m_net->w, m_net->h);

		layer l = m_net->layers[m_net->n - 1];

		float* X = sized.data;

		network_predict(*m_net, X);
		
		int nboxes = 0;
		detection* dets = get_network_boxes(m_net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, /*letter_box*/0);
		float nms = .45;    // 0.4F
		if (nms) {
			if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
			else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
		}

		draw_detections_v3_custom(im, dets, nboxes, thresh, m_names, /*alphabet,*/ l.classes, /*ext_output*/0, my_data);

		//my_data
		my_data.frame = image_to_mat(im);

		//printf("%f / %f / %f / %f\n", my_data.boxes.at(0).x, my_data.boxes.at(0).y, my_data.boxes.at(0).w, my_data.boxes.at(0).h);

		free_detections(dets, nboxes);
		free_image(im);
		free_image(sized);	
	}
	catch (int expn)
	{
		std::cout << "[error] : " << expn << std::endl;
	}
	return my_data;
}


void view(cv::Mat m)
{
	cv::imshow("tt", m);
	cv::waitKey(1); //0=무한
}


My_Data Frame(MY_QUEUE *my_queue, cv::Mat m)
{	
	MY_ALGORITHM* my_algorithm = MY_ALGORITHM::get_instance();
	My_Data my_data = my_algorithm->Detect_Image(m);

	my_queue->Put_All(my_data);

	return my_data;
}

void MY_ALGORITHM::DetectVideo(char * video_file)
{
	cv::VideoCapture cap;
	bool isOpen;
	cv::Mat returnMat;

	if (!strcmp(video_file,"CAM")) //확인필요
	{
		std::cout << "Reading From CAM : 0" << std::endl;
		
		for (int i = 0; i < 10; i++)
		{
			isOpen = cap.open(0);
			if (isOpen) break;
		}
	}
	else
	{
		std::cout << "Reading From VIDEO : "<< video_file << std::endl;
		
		for (int i = 0; i < 10; i++)
		{
			isOpen = cap.open(video_file);
			if (isOpen) break;
		}
	}
	if (!isOpen)
	{
		std::cout << "Failed to Read" << std::endl;
		return;
	}

	MY_QUEUE* my_queue = my_queue->get_instance();

	cv::Mat frame;

	my_queue->stop = false;

#ifdef SAVE_FRAME
	int i = 0;
#endif

	while (cap.grab())
	{		
		cap.retrieve(frame);

#ifdef SAVE_FRAME
		i++;
		if (i % 50 == 0)
		{
			char name[256] = { 0 };
			sprintf(name, "save\\%d.jpg", i);
			cv::imwrite(name,frame);
		}
#endif
		Frame(my_queue, frame);
	}
	my_queue->stop = true;	
}



//IOU-----------------------------

float GetIOU(iou_box box1, iou_box box2)
{//float left_x, right_x, top_y, bot_y;
	float mlX = std::max(box1.left_x, box2.left_x);
	float mrX = std::min(box1.right_x, box2.right_x);
	float mtY = std::max(box1.top_y, box2.top_y);
	float mbY = std::min(box1.bot_y, box2.bot_y);

	float interArea = std::max(0.0f, mrX - mlX + 1) * std::max(0.0f, mbY - mtY + 1);

	float box1Area = (box1.right_x - box1.left_x + 1) * (box1.bot_y - box1.top_y + 1);
	float box2Area = (box2.right_x - box2.left_x + 1) * (box2.bot_y - box2.top_y + 1);

	float iou = interArea / (box1Area + box2Area - interArea);

	return iou;
}


void PredictedType(std::vector<iou_box> boxes1, std::vector<iou_box> boxes2, unsigned int pType[4])
{ 
	//정렬이 제대로 되어있는 조건이어야 합니다!!		
	//기준 값은 정하기 나름입니다.
	float threshold = 0.8f;

	//boxes1을 정답 박스로 간주합니다!!

    //정답 박스가 예측 박스보다 많을 경우 = 예측을 다못한 경우
	if (boxes1.size() > boxes2.size())
	{   
		//False negative 실제 결과는 참인데, 거짓으로 추론함 (추론하지 못함)
		pType[FN] += boxes1.size() - boxes2.size();

        //예측한 박스들 중에서도 잘못 된 값인지 확인
		for (int i = 0; i < boxes2.size(); i++)
		{
			if (GetIOU(boxes1[i], boxes2[i]) >= threshold)
			{
				pType[TP] += 1;
			}
			else pType[FP] += 1;
		}
	}
    //예측한 박스가 정답박스보다 많을 경우
	else if (boxes1.size() < boxes2.size())
	{
		pType[FP] += boxes2.size() - boxes1.size();

		for (int i = 0; i < boxes1.size(); i++)
		{
			if (GetIOU(boxes1[i], boxes2[i]) >= threshold)
			{
				pType[TP] += 1;
			}
			else pType[FP] += 1;
		}
	}
	else
	{   //동일한 값일 경우
		// 만약 둘 다 0,0,0,0이면 TN = 옳게 검출되지 않은 값
		// 혹은 IOU로 확인
		for (int i = 0; i < boxes1.size(); i++)
		{
			if (((boxes1[i].left_x == 0.0f) && (boxes1[i].right_x == 0.0f) && (boxes1[i].top_y == 0.0f) && (boxes1[i].bot_y == 0.0f))
				&& ((boxes2[i].left_x == 0.0f) && (boxes2[i].right_x == 0.0f) && (boxes2[i].top_y == 0.0f) && (boxes2[i].bot_y == 0.0f)))
			{
				pType[TN] += 1;
				continue;
			}

			if (GetIOU(boxes1[i], boxes2[i]) >= threshold)
			{
				pType[TP] += 1;
			}
			else pType[FP] += 1;
		}
	}
		
	//accuracy = (pType[TP] + pType[TN]) / (pType[TP] + pType[TN] + pType[FP] + pType[FN]);
	return;
}


void MY_ALGORITHM::ReadBoxfile(char* boxfile1, char* boxfile2)
{
	iou_box boxes1[125][2] = { 0, }; //야매로 하자 야매로
	iou_box boxes2[125][2] = { 0, };

	char* pch;

	FILE* file = fopen(boxfile1, "r");

	int i = 0;

	if (file != NULL) {
		char line[256] = { 0 };
		while (fgets(line, sizeof line, file) != NULL) /* read a line from a file */ 
		{
			//fprintf(stdout, "%s", line); //print the file contents on stdout.

			int j = 0, k = 0;

			pch = strtok(line, ",");
			while (pch != NULL) {
				printf("%s\n", pch);

				switch (k)
				{
					case 0:
						boxes1[i][j].left_x = atoi(pch);
						k+=1;
						break;
					case 1:
						boxes1[i][j].right_x = atoi(pch);
						k += 1;
						break;
					case 2:
						boxes1[i][j].top_y = atoi(pch);
						k += 1;
						break;
					case 3:
						boxes1[i][j].bot_y = atoi(pch);
						j += 1;
						k = 0;
						break;
					default:
						break;
				}				

				pch = strtok(NULL, ",");
			}

			i += 1;
		}

		fclose(file);
	}
	else {
		AfxMessageBox(_T("boxfile 1")); //print the error message on stderr.
	}


	i = 0;
	file = fopen(boxfile2, "r");
	if (file != NULL) {
		char line[256] = { 0 };
		while (fgets(line, sizeof line, file) != NULL) /* read a line from a file */
		{
			//fprintf(stdout, "%s", line); //print the file contents on stdout.

			int j = 0, k = 0;

			pch = strtok(line, ",");
			while (pch != NULL) {
				printf("%s\n", pch);

				switch (k)
				{
				case 0:
					boxes2[i][j].left_x = atoi(pch);
					k += 1;
					break;
				case 1:
					boxes2[i][j].right_x = atoi(pch);
					k += 1;
					break;
				case 2:
					boxes2[i][j].top_y = atoi(pch);
					k += 1;
					break;
				case 3:
					boxes2[i][j].bot_y = atoi(pch);
					j += 1;
					k = 0;
					break;
				default:
					break;
				}

				pch = strtok(NULL, ",");
			}

			i += 1;
		}

		fclose(file);
	}
	else {
		AfxMessageBox(_T("boxfile 1")); //print the error message on stderr.
	}

	unsigned int pType[4] = { 0 };

	for (int a = 0; a < 125; a++)
	{
		std::vector<iou_box> vboxes1, vboxes2;

		for (int b = 0; b < 2; b++)
		{			
			if ((b>0)&&(boxes1[a][b].left_x == 0.0f)) continue;
			vboxes1.push_back(boxes1[a][b]);
			if ((b > 0) && (boxes2[a][b].left_x == 0.0f)) continue;
			vboxes2.push_back(boxes2[a][b]);
		}

		unsigned int tType[4] = { 0 };
		PredictedType(vboxes1, vboxes2,tType);
		for (int c = TP; c < TN+1; c++)
		{
			//unsigned int tValue = 0;
			//memcpy(&tValue, &tType[c], sizeof(unsigned int));
			//pType[c] += tValue;
			pType[c] += tType[c];
		}

	}

	float accuracy = (float)(pType[TP] + pType[TN]) / (float)(pType[TP] + pType[TN] + pType[FP] + pType[FN]);

	printf("accuracy = %f", accuracy);

	return;
}


std::vector<iou_box> MY_ALGORITHM::Detect_Image_for_IOU(cv::Mat f)
{
	std::vector<iou_box> my_boxes;
	try {
		//thresh, hier_thresh
		float thresh = 0.25, hier_thresh = 0.5;

		CV_Assert(!f.empty()); // empty일 경우 Error , 내부가 false일 경우

		image im = mat_to_image(f);

		image sized = resize_image(im, m_net->w, m_net->h);

		layer l = m_net->layers[m_net->n - 1];

		float* X = sized.data;

		network_predict(*m_net, X);

		int nboxes = 0;
		detection* dets = get_network_boxes(m_net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, /*letter_box*/0);
		float nms = .45;    // 0.4F
		if (nms) {
			if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
			else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
		}

		draw_detections_v3_custom(im, dets, nboxes, thresh, m_names, /*alphabet,*/ l.classes, /*ext_output*/0, my_boxes);
			
		free_detections(dets, nboxes);
		free_image(im);
		free_image(sized);
	}
	catch (int expn)
	{
		std::cout << "[error] : " << expn << std::endl;
	}
	return my_boxes;
}


void MY_ALGORITHM::DetectFolder(char * folder)
{
	CStringList imageList;
	//폴더 내 파일 읽음
	CFileFind fileFinder;

	CString cstrFolder(folder);
	cstrFolder += _T("\\*.*");

	//Train 폴더 먼저 읽기
	bool bExist = fileFinder.FindFile(cstrFolder);
	while (bExist)
	{
		//다음 파일이 있을 경우 TRUE
		bExist = fileFinder.FindNextFileW();

		CString fileName = fileFinder.GetFileName();

		if (fileName == _T(".") || fileName == _T("..") || fileName == _T("Thumbs.db"))
			continue;

		fileName = fileFinder.GetFilePath();
		imageList.AddTail(fileName);
	}

	//읽은 값을 txt로 저장
	CFile fileSave;

	//Train
	if (!fileSave.Open(_T("iou_box.txt"), CFile::modeCreate | CFile::modeWrite))
	{
		AfxMessageBox(TEXT("iou_box.txt를 생성하지 못했습니다."));
	}


	POSITION pos = imageList.GetHeadPosition();
	for (int i = 0; i < imageList.GetCount(); i++)
	{
		std::vector<iou_box> my_boxes;

		//my_box = { 0.0f, 0.0f, 0.0f, 0.0f };

		std::string getImagePath = CT2CA(imageList.GetNext(pos));
		cv::Mat img = cv::imread(getImagePath);
		my_boxes = Detect_Image_for_IOU(img);
		
		char cmy_box[256] = { 0 };

		if(my_boxes.size()==0) sprintf(cmy_box, "%f,%f,%f,%f", 0.0f, 0.0f, 0.0f, 0.0f);		
		else 
		{
			for (int j = 0; j < my_boxes.size(); j++)
			{
				if (j > 0)	cmy_box[strlen(cmy_box) - 1] = ',';
				sprintf(cmy_box, "%s%f,%f,%f,%f", cmy_box,my_boxes[j].left_x, my_boxes[j].right_x, my_boxes[j].top_y, my_boxes[j].bot_y);				
			}
		}
		cmy_box[strlen(cmy_box) - 1] = '\n';
		fileSave.Write(cmy_box, strlen(cmy_box));
	}
	fileSave.Close();
}
