#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "std_msgs/msg/header.hpp"
#include "object_detection/msg/detection.hpp"
#include "object_detection/msg/detection_array.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"


class VisualizerNode : public rclcpp::Node {
    public:
        // Define a synchronization policy for the message filters
        using ImageMsg = sensor_msgs::msg::Image;
        using DetectionArrayMsg = object_detection::msg::DetectionArray;
        using SyncPolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, DetectionArrayMsg>;

        VisualizerNode() : Node("visualizer_node") {
            RCLCPP_INFO(this->get_logger(), "Starting Visualizer Node...");

            image_sub_.subscribe(this, "/video_stream");
            detections_sub_.subscribe(this, "/detections");

            sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), image_sub_, detections_sub_);
            sync_->registerCallback(std::bind(&VisualizerNode::sync_callback, this, std::placeholders::_1, std::placeholders::_2));

            RCLCPP_INFO(this->get_logger(), "Visualizer Node initialized and subscribed to video_stream.");
        }
    private:
        message_filters::Subscriber<ImageMsg> image_sub_;
        message_filters::Subscriber<DetectionArrayMsg> detections_sub_;
        std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

        void sync_callback(const ImageMsg::ConstSharedPtr& img_msg, const DetectionArrayMsg::ConstSharedPtr& det_msg) {
            // Convert ROS image message to OpenCV image
            cv_bridge::CvImagePtr cv_ptr;
            cv::Mat frame;
            try {
                cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
                frame = cv_ptr->image;
            } catch (cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }

            for (const auto& detection : det_msg->detections) {
                cv::rectangle(
                    frame,
                    cv::Point(static_cast<int>(detection.x), static_cast<int>(detection.y)),
                    cv::Point(static_cast<int>(detection.x + detection.width), static_cast<int>(detection.y + detection.height)),
                    cv::Scalar(0, 255, 0),
                    2
                );
                cv::putText(
                    frame,
                    detection.label,
                    cv::Point(static_cast<int>(detection.x), static_cast<int>(detection.y) - 10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 255, 0),
                    2
                );
            }

            cv::imshow("Detections", frame);
            cv::waitKey(1);
        }
};


int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisualizerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}