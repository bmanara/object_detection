#include <memory>
#include <string>

#include <torch/script.h>
#include <torch/torch.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>


class OwlViTRosNode : public rclcpp::Node {
    public:
        OwlViTRosNode() : Node("owl_vit_ros_node") {
            RCLCPP_INFO(this->get_logger(), "Starting OwlViT ROS Node...");
            this->load_model("/home/bmacraze/ros2_kilted_ws/src/object_detection/models/owlvit_onnx/model.onnx");
        }
    
    private:
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

        std::vector<int64_t> cached_input_ids_;
        std::vector<int64_t> cached_attention_mask_;
        const int max_queries = 16;
        const int max_tokens = 16;

        Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "OwlViT"};
        std::unique_ptr<Ort::Session> session_;
        Ort::MemoryInfo memory_info_{nullptr};

        void load_model(const std::string& model_path) {
            RCLCPP_INFO(this->get_logger(), "Loading ONNX model from: %s", model_path.c_str());
            env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "OwlViTEnv");
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(4);
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
            memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            RCLCPP_INFO(this->get_logger(), "ONNX model loaded successfully from: %s", model_path.c_str());
        }
};


int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OwlViTRosNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}