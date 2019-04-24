#include "deeplcd.h"

#include <dirent.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

/*************************************************************************
* Here is quick cpu demo using a few images from the Kitti odometry dataset
**************************************************************************/ 

int main(int argc, char** argv)
{
	
	std::cout << "Deep Loop Closure Detection Demo!\n";

	// std::vector<std::string> fls = {};
	struct dirent *ptr;    
	struct dirent *ptr1;
	DIR *dir1;
    DIR *dir;
	std::string PATH = "./images/live/";
	std::string PATH1 = "./images/memory/";
	dir=opendir(PATH.c_str());
	dir1 = opendir(PATH1.c_str());
    std::vector<std::string> fls;
	std::vector<std::string> fls1;
    while((ptr=readdir(dir))!=NULL)
    {
 
        if(ptr->d_name[0] == '.')
            continue;
        //cout << ptr->d_name << endl;
        fls.push_back(ptr->d_name);
    }
	while ((ptr1=readdir(dir1))!=NULL)
	{
		if(ptr1->d_name[0] == '.')
            continue;
        //cout << ptr->d_name << endl;
        fls1.push_back(ptr1->d_name);
	}

	std::string images = "images/";
	std::string live = "live/";
	std::string mem = "memory/";

	deeplcd::DeepLCD test_lcd; // Using default constructor, takes net from rvbaa_model directory downloaded on compilation

	cv::Mat im;

	cv::Size sz(160, 120);
	
	std::string curr;
	
	for (std::string fl : fls1)
	{	
		curr = images + mem + fl;
		// std::cout << "Loading image " << curr << "\n";
		im = cv::imread(curr);
		cv::cvtColor(im, im, cv::COLOR_BGR2GRAY); // convert to grayscale
		cv::resize(im, im, sz);
		// std::cout << "Added Image " << test_lcd.add(im) << " to database\n";
		test_lcd.add(im);
	}
	std::cout << "Loaded Images Successfully" <<"\n";
	std::cout << "\n------------------------------------------\n";
	cv::Mat score_mat;
	score_mat.create(fls.size(),fls1.size(),5);
	// Okay now we have a database of descriptors, lets see if we can match them now
	std::cerr << "\nWriting output...\n";

 	std::string output_path("confusion_matrix.txt");
  	std::ofstream of;
  	of.open(output_path);
  	if (of.fail()) {
    	std::cerr << "Failed to open output file " << output_path << std::endl;
   		exit(1);
  	}

	int i = 0;
	// deeplcd::query_result q;
	for (std::string fl : fls)
	{	
		curr = images + live + fl;
		// std::cout << "\nLoading image " << curr << " for database query\n";
		im = cv::imread(curr);
		cv::cvtColor(im, im, cv::COLOR_BGR2GRAY); // convert to grayscale
		cv::resize(im, im, sz);
		for (int j=0; j < fls1.size();j++){
			deeplcd::descriptor descr_im = test_lcd.calcDescr(im);
			score_mat.at<float>(i,j) = test_lcd.score(descr_im.descr, test_lcd.db[j].descr);
			std::cout << "The score is: " <<score_mat.at<float>(i,j)<<std::endl;
		}
		i++;
		// q = test_lcd.query(im, 0); // query(im, false) means return 1 result in q, and don't add im's descriptor to database afterwards
		// if (q.score > 0){
			// std::cout << "Image " << ++i << " result: " << q << "\n";
		// }

	}
	for (int i = 0; i < fls.size(); i++) {
    	for (int j = 0; j < fls1.size(); j++) { 
			of << score_mat.at<float>(i,j) <<" ";
		}
		of << "\n"; 
	}
	std::cerr<< "Output done\n";
}

