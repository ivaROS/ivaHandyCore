#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//including "linemod.cpp"s including files
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h> // cvFindContours
#include <opencv2/objdetect/objdetect.hpp>
#include <iterator>
#include <set>
#include <cstdio>
#include <iostream>

#include "linemodlib.cpp"

//for subscribing multiple topics
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <image_transport/subscriber_filter.h>//for subscribing multiple topics incluing image
#include <message_filters/sync_policies/approximate_time.h>


// Function prototypes
void subtractPlane(const cv::Mat& depth, cv::Mat& mask, std::vector<CvPoint>& chain, double f);

std::vector<CvPoint> maskFromTemplate(const std::vector<cv::linemod::Template>& templates,
                                      int num_modalities, cv::Point offset, cv::Size size,
                                      cv::Mat& mask, cv::Mat& dst);

void templateConvexHull(const std::vector<cv::linemod::Template>& templates,
                        int num_modalities, cv::Point offset, cv::Size size,
                        cv::Mat& dst);

void drawResponse(const std::vector<cv::linemod::Template>& templates,
                  int num_modalities, cv::Mat& dst, cv::Point offset, int T);

cv::Mat displayQuantized(const cv::Mat& quantized);

// Copy of cv_mouse from cv_utilities
class Mouse
{
public:
  static void start(const std::string& a_img_name)
  {
    cvSetMouseCallback(a_img_name.c_str(), Mouse::cv_on_mouse, 0);
  }
  static int event(void)
  {
    int l_event = m_event;
    m_event = -1;
    return l_event;
  }
  static int x(void)
  {
    return m_x;
  }
  static int y(void)
  {
    return m_y;
  }

private:
  static void cv_on_mouse(int a_event, int a_x, int a_y, int, void *)
  {
    m_event = a_event;
    m_x = a_x;
    m_y = a_y;
  }

  static int m_event;
  static int m_x;
  static int m_y;
};
int Mouse::m_event;
int Mouse::m_x;
int Mouse::m_y;

static void help()
{
  printf("Usage: openni_demo [templates.yml]\n\n"
         "Place your object on a planar, featureless surface. With the mouse,\n"
         "frame it in the 'color' window and right click to learn a first template.\n"
         "Then press 'l' to enter online learning mode, and move the camera around.\n"
         "When the match score falls between 90-95%% the demo will add a new template.\n\n"
         "Keys:\n"
         "\t h   -- This help page\n"
         "\t l   -- Toggle online learning\n"
         "\t m   -- Toggle printing match result\n"
         "\t t   -- Toggle printing timings\n"
         "\t w   -- Write learned templates to disk\n"
         "\t [ ] -- Adjust matching threshold: '[' down,  ']' up\n"
         "\t q   -- Quit\n\n");
}

// Adapted from cv_timer in cv_utilities
class Timer
{
public:
  Timer() : start_(0), time_(0) {}

  void start()
  {
    start_ = cv::getTickCount();
  }

  void stop()
  {
    CV_Assert(start_ != 0);
    int64 end = cv::getTickCount();
    time_ += end - start_;
    start_ = 0;
  }

  double time()
  {
    double ret = time_ / cv::getTickFrequency();
    time_ = 0;
    return ret;
  }

private:
  int64 start_, time_;
};

// Functions to store detector and templates in single XML/YAML file
static cv::Ptr<cv::linemod::Detector> readLinemod(const std::string& filename)
{
  cv::Ptr<cv::linemod::Detector> detector = new cv::linemod::Detector;
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  detector->read(fs.root());

  cv::FileNode fn = fs["classes"];
  for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
    detector->readClass(*i);

  return detector;
}

static void writeLinemod(const cv::Ptr<cv::linemod::Detector>& detector, const std::string& filename)
{
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  detector->write(fs);

  std::vector<std::string> ids = detector->classIds();
  fs << "classes" << "[";
  for (int i = 0; i < (int)ids.size(); ++i)
  {
    fs << "{";
    detector->writeClass(ids[i], fs);
    fs << "}"; // current class
  }
  fs << "]"; // classes
}


static const std::string OPENCV_WINDOW = "Image window";
static const std::string OPENCV_WINDOW1 = "Image window1";

class ImageConverter {
public:
  ImageConverter() :
    it_(nh_),
    color_image_sub_( it_, "/camera/rgb/image_color", 1 ),
    depth_image_sub_( it_, "/camera/depth/image_raw", 1 ),
    roi_size(200,200),

    sync( MySyncPolicy( 10 ), color_image_sub_, depth_image_sub_ )
  {

    // Various settings and flags
    show_match_result = true;
    show_timings = false;
    learn_online = false;
    num_classes = 0;
    matching_threshold = 80;
    /// @todo Keys for changing these?
    //roi_size = (200, 200);
    learning_lower_bound = 90;
    learning_upper_bound = 95;

    // Initialize HighGUI
    help();
    cv::namedWindow("color");
    cv::namedWindow("normals");
    Mouse::start("color");

    filename = "linemod_templates_ball.yml";
    detector = readLinemod(filename);

    ids = detector->classIds();
    num_classes = detector->numClasses();

    printf("Loaded %s with %d classes and %d templates\n",
           "linemod_templates.yml", num_classes, detector->numTemplates());
    if (!ids.empty())
    {
      printf("Class ids:\n");
      std::copy(ids.begin(), ids.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
    }

    num_modalities = (int)detector->getModalities().size();
    focal_length = 575.000;//TO DO.. should be modified by case;



    sync.registerCallback( boost::bind( &ImageConverter::callback, this, _1, _2 ) );
    image_pub_ = it_.advertise("/image_converter/output_video", 1);
    cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }




//CallBack funstion starts here..//
  void callback( const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::ImageConstPtr& msg1) {  

//Get color image//  
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat color = cv_ptr->image;

    // Draw an example circle on the video stream
    if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
      cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, color);

//Get depth image// 
    cv_bridge::CvImagePtr cv_ptr1;
    try
    {
      cv_ptr1 = cv_bridge::toCvCopy(msg1, sensor_msgs::image_encodings::TYPE_16UC1);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat depth = cv_ptr1->image;
    // Draw an example circle on the video stream
    if (cv_ptr1->image.rows > 60 && cv_ptr1->image.cols > 60)
      cv::circle(cv_ptr1->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW1, depth);























/*
////test read file///
const std::string filename = "linemod_templates.yml";
cv::FileStorage fs( filename, cv::FileStorage::READ);


std::cout<<"is open? "<<fs.isOpened()<<std::endl;
int pyramid_levels = fs["pyramid_levels"];
std::cout<<pyramid_levels<<std::endl;
std::string T_at_level;
fs["T"] >> T_at_level;
std::cout<<T_at_level<<std::endl;
*/








    cv::waitKey(3);
    
    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }

private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;

  typedef image_transport::SubscriberFilter ImageSubscriber;


  ImageSubscriber color_image_sub_;
  ImageSubscriber depth_image_sub_;

  typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image, sensor_msgs::Image
  > MySyncPolicy;

  message_filters::Synchronizer< MySyncPolicy > sync;

  image_transport::Publisher image_pub_;

  //members used in callback function (originally in main function in linemod.cpp) 
  // Various settings and flags
  bool show_match_result;
  bool show_timings;
  bool learn_online;
  int num_classes;
  int matching_threshold;
  /// @todo Keys for changing these?
  cv::Size roi_size;
  int learning_lower_bound;
  int learning_upper_bound;

  // Timers
  Timer extract_timer;
  Timer match_timer;

  // Initialize LINEMOD data structures
  cv::Ptr<cv::linemod::Detector> detector;
  std::string filename;

  std::vector<std::string> ids;
  int num_modalities;
  double focal_length;


};

int main(int argc, char** argv) {
  ros::init( argc, argv, "image_converter" );
  ImageConverter ic;

  //while( ros::ok() ){
    ros::spin();
  //}

  return 0;
}
 
