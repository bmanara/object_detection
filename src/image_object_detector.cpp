#include <memory>
#include <string>

#include <torch/script.h>
#include <torch/torch.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <tokenizers_cpp.h>


#define MODEL_NAME "owlvit_onnx"

using tokenizers::Tokenizer;

std::string LoadBytesFromFile(const std::string& path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
}


class OwlViTRosNode : public rclcpp::Node {
    public:
        OwlViTRosNode() : Node("owl_vit_ros_node") {
            RCLCPP_INFO(this->get_logger(), "Starting OwlViT ROS Node...");
            this->load_tokenizer(MODEL_NAME);
            this->load_model(MODEL_NAME);
            this->process_text_prompts(cached_input_ids_, cached_attention_mask_);
        }
    
    private:
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

        // Cached tokenized inputs for the text queries
        std::vector<int64_t> cached_input_ids_;
        std::vector<int64_t> cached_attention_mask_;
        const int max_queries = 16;
        const int max_tokens = 16;
        std::vector<std::string> text_queries_ = {
            "cat", "dog", "car", "remote controller", "cat tail"
        };

        // Tokenizer for text processing
        std::unique_ptr<Tokenizer> tokenizer_;

        // ONNX Runtime environment and session
        Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "OwlViT"};
        std::unique_ptr<Ort::Session> session_;
        Ort::MemoryInfo memory_info_{nullptr};

        void load_tokenizer(const std::string& model_name) {
            std::string tokenizer_path = "/home/bmacraze/ros2_kilted_ws/src/object_detection/models/" + model_name + "/tokenizer.json";
            RCLCPP_INFO(this->get_logger(), "Loading tokenizer from: %s", tokenizer_path.c_str());
            auto blob = LoadBytesFromFile(tokenizer_path);
            tokenizer_ = Tokenizer::FromBlobJSON(blob);
            RCLCPP_INFO(this->get_logger(), "Tokenizer loaded successfully from: %s", tokenizer_path.c_str());
        }

        void load_model(const std::string& model_name) {
            std::string model_path = "/home/bmacraze/ros2_kilted_ws/src/object_detection/models/" + model_name + "/model.onnx";
            RCLCPP_INFO(this->get_logger(), "Loading ONNX model from: %s", model_path.c_str());
            env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "OwlViTEnv");
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(4);
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
            memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            RCLCPP_INFO(this->get_logger(), "ONNX model loaded successfully from: %s", model_path.c_str());
        }

        void process_text_prompts(std::vector<int64_t>& input_ids, std::vector<int64_t>& attention_mask) {
            RCLCPP_INFO(this->get_logger(), "Processing text prompts for ONNX inference...");
            input_ids.assign(1 * max_queries * max_tokens, 0);
            attention_mask.assign(1 * max_queries * max_tokens, 0);

            for(size_t q = 0; q < text_queries_.size() && q < static_cast<size_t>(max_queries); ++q) {
                auto encoding = tokenizer_->Encode(text_queries_[q]);
                std::vector<int32_t> ids = encoding;

                for(size_t t = 0; t < ids.size() && t < static_cast<size_t>(max_tokens); ++t) {
                    size_t flat_idx = (q * max_tokens) + t;
                    input_ids[flat_idx] = static_cast<int64_t>(ids[t]);
                    attention_mask[flat_idx] = 1; // Mark active tokens
                }
            }
            RCLCPP_INFO(this->get_logger(), "Text prompts processed and cached for ONNX inference.");
        }

        void image_callback(const std::string& image_path) {
            RCLCPP_INFO(this->get_logger(), "Processing image: %s", image_path.c_str());
            cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
            if (img.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to read image from: %s", image_path.c_str());
                return;
            }
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