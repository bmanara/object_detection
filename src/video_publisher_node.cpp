#include <string>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/opencv.hpp>


using namespace std::chrono_literals;

class VideoPublisherNode : public rclcpp::Node {
    public:
        VideoPublisherNode() : Node("video_publisher_node") {
            RCLCPP_INFO(this->get_logger(), "Starting Video Publisher Node...");
            
            image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("video_stream", 10);

            // Either 0 or link to a video file
            cap_.open(0); // Open the default camera (index 0)
            
            if (!cap_.isOpened()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open video capture device.");
                return;
            }

            RCLCPP_INFO(this->get_logger(), "Video capture device opened successfully. Starting to publish video frames...");
            timer_ = this->create_wall_timer(33ms, std::bind(&VideoPublisherNode::timer_callback, this));
        }

        ~VideoPublisherNode() {
            if (cap_.isOpened()) {
                cap_.release();
            }
        }

    private:
        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
        cv::VideoCapture cap_;

        void timer_callback() {
            cv::Mat frame;
            bool ret = cap_.read(frame); // Capture a new frame

            if (!ret || frame.empty()) {
                RCLCPP_WARN(this->get_logger(), "Failed to capture video frame.");
                return;
            }

            std::shared_ptr<sensor_msgs::msg::Image> msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
            msg->header.stamp = this->get_clock()->now();
            msg->header.frame_id = "camera_frame";

            image_pub_->publish(*msg);
        }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VideoPublisherNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}