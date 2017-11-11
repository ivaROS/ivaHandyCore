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

//including "linemodlib.cpp"s including files
#include "precomp.hpp"
#include <limits>


//for subscribing multiple topics
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <image_transport/subscriber_filter.h>//for subscribing multiple topics incluing image
#include <message_filters/sync_policies/approximate_time.h>


static const std::string OPENCV_WINDOW = "Image window";
static const std::string OPENCV_WINDOW1 = "Image window1";

class ImageConverter {
public:
  ImageConverter() :
    it_(nh_),
    color_image_sub_( it_, "/camera/rgb/image_color", 1 ),
    depth_image_sub_( it_, "/camera/depth/image_raw", 1 ),

    sync( MySyncPolicy( 10 ), color_image_sub_, depth_image_sub_ )
  {
    sync.registerCallback( boost::bind( &ImageConverter::callback, this, _1, _2 ) );
    image_pub_ = it_.advertise("/image_converter/output_video", 1);
    cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void callback( const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::ImageConstPtr& msg1) {    
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

    // Draw an example circle on the video stream
    if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
      cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);

////test read file///
const std::string filename = "linemod_templates.yml";
cv::FileStorage fs( filename, cv::FileStorage::READ);


std::cout<<"is open? "<<fs.isOpened()<<std::endl;
int pyramid_levels = fs["pyramid_levels"];
std::cout<<pyramid_levels<<std::endl;
std::string T_at_level;
fs["T"] >> T_at_level;
std::cout<<T_at_level<<std::endl;


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

    // Draw an example circle on the video stream
    if (cv_ptr1->image.rows > 60 && cv_ptr1->image.cols > 60)
      cv::circle(cv_ptr1->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW1, cv_ptr1->image);












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
};

int main(int argc, char** argv) {
  ros::init( argc, argv, "image_converter" );
  ImageConverter ic;

  //while( ros::ok() ){
    ros::spin();
  //}

  return 0;
}
 
