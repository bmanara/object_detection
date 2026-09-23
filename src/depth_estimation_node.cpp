#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "object_detection/msg/depth_map.hpp"


#define MODEL_NAME "depth_anything_v2_small_indoor"


class DepthEstimationNode : public rclcpp::Node {
public:
    DepthEstimationNode() : Node("depth_estimation_node") {
        RCLCPP_INFO(this->get_logger(), "Starting Depth Estimation Node...");
        this->load_model(MODEL_NAME);

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "video_stream", 
            10, 
            std::bind(&DepthEstimationNode::image_callback, this, std::placeholders::_1)
        );

        depth_pub_ = this->create_publisher<object_detection::msg::DepthMap>(
            "depth_map", 
            10
        );
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<object_detection::msg::DepthMap>::SharedPtr depth_pub_;

    // ONNX Runtime environment and session
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "DepthEstimation"};
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_{nullptr};

    void load_model(const std::string& model_name) {
        std::string model_path = "/home/bmacraze/ros2_kilted_ws/src/object_detection/models/" + model_name + "/model.onnx";
        RCLCPP_INFO(this->get_logger(), "Loading ONNX model from: %s", model_path.c_str());
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "DepthEstimationEnv");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
        memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        RCLCPP_INFO(this->get_logger(), "ONNX model loaded successfully from: %s", model_path.c_str());
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImageConstPtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat img = cv_ptr->image.clone();
        if (img.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Received empty image frame.");
            return;
        }

        // Following DINOv2 architecture
        int target_h = 518;
        int target_w = 518;

        std::vector<float> input_tensor_values = preprocess_image(img, target_h, target_w);

        std::vector<int64_t> input_shape = {1, 3, target_h, target_w};
        std::vector<int64_t> output_shape = {1, target_h, target_w};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size()
        );

        // Must match export script written in python
        const char* input_names[] = {"pixel_values"};
        const char* output_names[] = {"depth"};

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, 
            input_names, 
            &input_tensor, 
            1, 
            output_names, 
            1
        );

        RCLCPP_DEBUG(this->get_logger(), "Inference completed. Output tensor size: %zu", output_tensors.size());
        
        float* output_data = output_tensors[0].GetTensorMutableData<float>();

        cv::Mat depth_model_res(target_h, target_w, CV_32FC1, output_data);

        cv::Mat depth_orig_res;
        cv::Size orig_size(img.cols, img.rows);

        cv::resize(depth_model_res, depth_orig_res, orig_size);

        auto depth_msg = object_detection::msg::DepthMap();
        depth_msg.header.stamp = this->now();
        depth_msg.header.frame_id = "depth_frame";
        depth_msg.height = orig_size.height;
        depth_msg.width = orig_size.width;
        depth_msg.k = {
            double(orig_size.width), 0.0, double(orig_size.width) / 2.0,
            0.0, double(orig_size.height), double(orig_size.height) / 2.0,
            0.0, 0.0, 1.0
        };

        size_t total_pixels = orig_size.width * orig_size.height;
        float* depth_data_ptr = reinterpret_cast<float*>(depth_orig_res.data);
        depth_msg.depth_data.assign(depth_data_ptr, depth_data_ptr + total_pixels);
        depth_msg.min_depth = *std::min_element(depth_msg.depth_data.begin(), depth_msg.depth_data.end());
        depth_msg.max_depth = *std::max_element(depth_msg.depth_data.begin(), depth_msg.depth_data.end());

        depth_pub_->publish(depth_msg);

        return;
    }

    std::vector<float> preprocess_image(const cv::Mat& img, int target_h, int target_w) {
        cv::Mat rgb, resized, float_img;
        
        // Convert from RGB to BGR
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

        // Resize the image to the target dimensions
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(target_w, target_h), 0, 0, cv::INTER_CUBIC);

        // Convert to float and normalize
        resized_img.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

        // Normalize using ImageNet mean and std
        cv::Scalar mean(0.485, 0.456, 0.406);
        cv::Scalar std_dev(0.229, 0.224, 0.225);

        std::vector<float> tensor_data(1 * 3 * target_h * target_w);
        const int plane_size = target_h * target_w;

        // 4. Populate planar NCHW layout safely
        for (int h = 0; h < target_h; ++h) {
            for (int w = 0; w < target_w; ++w) {
                // Must use cv::Vec3f because float_img is CV_32FC3
                const cv::Vec3f& pixel = float_img.at<cv::Vec3f>(h, w);

                tensor_data[0 * plane_size + h * target_w + w] = (pixel[0] - mean[0]) / std_dev[0]; // R
                tensor_data[1 * plane_size + h * target_w + w] = (pixel[1] - mean[1]) / std_dev[1]; // G
                tensor_data[2 * plane_size + h * target_w + w] = (pixel[2] - mean[2]) / std_dev[2]; // B
            }
        }

        return tensor_data;
    }

    cv::Mat postprocess_depth(float* depth_data, int model_w, int model_h, const cv::Size& orig_size) {
        cv::Mat depth_metric(model_h, model_w, CV_32FC1, depth_data);

        cv::Mat depth_resized;
        cv::resize(depth_metric, depth_resized, orig_size);

        // TODO: Should be done in visualizer node... 
        // Convert to 8-bit heatmap for visualization
        double min_val, max_val;
        cv::minMaxLoc(depth_resized, &min_val, &max_val);

        cv::Mat depth_normalized, depth_colormap;
        // Map values to 0-255 range
        depth_resized.convertTo(depth_normalized, CV_8UC1, 255.0 / (max_val - min_val), -min_val * (255.0 / (max_val - min_val)));

        // Apply color map
        cv::applyColorMap(depth_normalized, depth_colormap, cv::COLORMAP_INFERNO);

        return depth_colormap;
    }
};


int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DepthEstimationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
