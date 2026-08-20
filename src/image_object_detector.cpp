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
#define PRED_THRESHOLD 0.5f

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
            this->image_callback("/home/bmacraze/ros2_kilted_ws/src/object_detection/images/room.jpeg");
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
            "table", "chair", "television", "cat", "window", "door", "person", "remote controller", "dog", "cat", "laptop", "keyboard", "mouse", "book", "cup", "bottle"
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
            const int64_t clip_sot_token_id = 49406;
            const int64_t clip_eot_token_id = 49407;
            input_ids.assign(1 * max_queries * max_tokens, clip_eot_token_id);
            attention_mask.assign(1 * max_queries * max_tokens, 0);

            for(size_t q = 0; q < text_queries_.size() && q < static_cast<size_t>(max_queries); ++q) {
                std::string prompt = "a photo of a " + text_queries_[q];
                auto encoding = tokenizer_->Encode(prompt);
                std::vector<int32_t> ids = encoding;

                size_t query_start_idx = q * max_tokens;
                size_t current_idx = query_start_idx;
                // input_ids[query_start_idx] = clip_sot_token_id; // Start of text
                // attention_mask[query_start_idx] = 1; // Mark active token

                input_ids[current_idx] = clip_sot_token_id; // Start of text
                attention_mask[current_idx] = 1; // Mark active token
                current_idx++;

                for(size_t t = 0; t < ids.size() && (current_idx - query_start_idx) < static_cast<size_t>(max_tokens - 1); ++t) {
                    // size_t flat_idx = (q * max_tokens) + t;
                    // input_ids[flat_idx] = static_cast<int64_t>(ids[t]);
                    // attention_mask[flat_idx] = 1; // Mark active tokens
                    if (ids[t] == clip_sot_token_id || ids[t] == clip_eot_token_id) {
                        RCLCPP_WARN(this->get_logger(), "Skipping special token in prompt '%s': token_id=%d", prompt.c_str(), ids[t]);
                        continue;
                    }

                    input_ids[current_idx] = static_cast<int64_t>(ids[t]);
                    attention_mask[current_idx] = 1; // Mark active tokens
                    current_idx++;
                }

                // size_t eot_idx = (q * max_tokens) + ids.size();
                // if (eot_idx < (q + 1) * max_tokens) {
                //     input_ids[eot_idx] = clip_eot_token_id; // End of text
                //     attention_mask[eot_idx] = 1; // Mark active token
                // }
                input_ids[current_idx] = clip_eot_token_id; // End of text
                attention_mask[current_idx] = 1; // Mark active token
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
            int orig_w = img.cols;
            int orig_h = img.rows;
            // Image Preprocessing: Resize and normalize
            // 768 x 768, ImageNet Mean/Std standard normalizations
            cv::Mat resized_img;
            cv::resize(img, resized_img, cv::Size(768, 768));
            // Convert BGR (OpenCV) to RGB expected by the model, then preprocess
            cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
            resized_img.convertTo(resized_img, CV_32FC3, 1.0 / 255.0);

            // cv::Scalar mean(0.485, 0.456, 0.406);
            // cv::Scalar std_dev(0.229, 0.224, 0.225);

            // Use CLIP Mean/Std, not ImageNet
            cv::Scalar mean(0.48145466, 0.4578275, 0.40821073);
            cv::Scalar std_dev(0.26862954, 0.26130258, 0.27577711);

            cv::Mat channels[3];
            cv::split(resized_img, channels);
            for (int i =0; i < 3; ++i) {
                channels[i] = (channels[i] - mean[i]) / std_dev[i];
            }
            cv::merge(channels, 3, resized_img);

            // Convert to CHW format for ONNX Runtime
            std::vector<float> input_tensor_values(1 * 3 * 768 * 768);
            std::vector<cv::Mat> chw_channels(3);
            for (int i = 0; i < 3; ++i) {
                chw_channels[i] = cv::Mat(768, 768, CV_32FC1, &input_tensor_values[i * 768 * 768]);
            }
            cv::split(resized_img, chw_channels);

            // Package everything into ONNX execution values
            std::vector<int64_t> img_shape = {1, 3, 768, 768};
            Ort::Value img_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, input_tensor_values.data(), input_tensor_values.size(), img_shape.data(), img_shape.size()
            );

            // Model expects 2D text tensors: {num_queries, num_tokens}
            std::vector<int64_t> text_shape = {static_cast<int64_t>(max_queries), static_cast<int64_t>(max_tokens)};
            Ort::Value id_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info_, cached_input_ids_.data(), cached_input_ids_.size(), text_shape.data(), text_shape.size()
            );

            Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info_, cached_attention_mask_.data(), cached_attention_mask_.size(), text_shape.data(), text_shape.size()
            );

            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(img_tensor));
            input_tensors.push_back(std::move(id_tensor));
            input_tensors.push_back(std::move(mask_tensor));

            const char* input_names[] = {"pixel_values", "input_ids", "attention_mask"};
            const char* output_names[] = {"logits", "pred_boxes"};

            // Run inference
            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr}, input_names, input_tensors.data(), input_tensors.size(),
                output_names, 2
            );

            // Process outputs
            float* logits = output_tensors[0].GetTensorMutableData<float>();
            float* pred_boxes = output_tensors[1].GetTensorMutableData<float>();

            auto logits_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            auto boxes_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

            // // Log output shapes for debugging
            // std::string lshape_str = "[";
            // for (size_t si = 0; si < logits_shape.size(); ++si) {
            //     lshape_str += std::to_string(logits_shape[si]);
            //     if (si + 1 < logits_shape.size()) lshape_str += ",";
            // }
            // lshape_str += "]";
            // std::string bshape_str = "[";
            // for (size_t si = 0; si < boxes_shape.size(); ++si) {
            //     bshape_str += std::to_string(boxes_shape[si]);
            //     if (si + 1 < boxes_shape.size()) bshape_str += ",";
            // }
            // bshape_str += "]";
            // RCLCPP_INFO(this->get_logger(), "Logits shape: %s, Boxes shape: %s", lshape_str.c_str(), bshape_str.c_str());

            int64_t num_boxes = 0;
            int64_t num_classes = 0;
            if (logits_shape.size() == 3) {
                num_boxes = logits_shape[1];
                num_classes = logits_shape[2];
            } else if (logits_shape.size() == 2) {
                num_boxes = logits_shape[0];
                num_classes = logits_shape[1];
            }

            // // Print a few sample logits for inspection
            // int64_t sample_count = std::min<int64_t>(10, num_boxes * std::max<int64_t>(1, num_classes));
            // for (int64_t s = 0; s < sample_count; ++s) {
            //     RCLCPP_INFO(this->get_logger(), "logit[%ld]=%f", s, logits[s]);
            // }

            // Sigmoid post-processing per box (assumes single best label per box)
            std::vector<std::tuple<float,int64_t,int64_t>> top_scores; // score, box, class
            for (int64_t i = 0; i < num_boxes; ++i) {
                // Compute sigmoid for each class
                for (int64_t j = 0; j < num_classes; ++j) {
                    float logit_value = logits[i * num_classes + j];
                    float score = 1.0f / (1.0f + exp(-logit_value)); // Sigmoid
                    top_scores.emplace_back(score, i, j);
                }
            }
            std::sort(top_scores.begin(), top_scores.end(), [](auto &a, auto &b){ return std::get<0>(a) > std::get<0>(b); });
            int topN = std::min<size_t>(5, top_scores.size());
            for (int t = 0; t < topN; ++t) {
                auto [score, bi, cj] = top_scores[t];
                float x_center = pred_boxes[bi * 4 + 0];
                float y_center = pred_boxes[bi * 4 + 1];
                float width = pred_boxes[bi * 4 + 2];
                float height = pred_boxes[bi * 4 + 3];
                // RCLCPP_INFO(this->get_logger(), "Top%d: class=%lld box=%lld sigmoid_score=%.6f box=[%.4f,%.4f,%.4f,%.4f]", t, (long long)cj, (long long)bi, score, x_center, y_center, width, height);
                // Draw boxes if above threshold and label within provided queries
                if (score > PRED_THRESHOLD && cj < static_cast<int64_t>(text_queries_.size())) {
                    float x1 = (x_center - width / 2.0f) * orig_w;
                    float y1 = (y_center - height / 2.0f) * orig_h;
                    float x2 = (x_center + width / 2.0f) * orig_w;
                    float y2 = (y_center + height / 2.0f) * orig_h;
                    cv::rectangle(img, cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                                  cv::Point(static_cast<int>(x2), static_cast<int>(y2)),
                                  cv::Scalar(0, 255, 0), 2);
                    cv::putText(img, text_queries_[cj], cv::Point(static_cast<int>(x1), static_cast<int>(y1) - 5),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                    RCLCPP_INFO(this->get_logger(), "Drew '%s' (class=%lld) with sigmoid score %.4f", text_queries_[cj].c_str(), (long long)cj, score);
                }
            }

            for (int64_t i = 0; i < num_boxes; ++i) {
                for (int64_t j = 0; j < num_classes; ++j) {
                    float logit_value = logits[i * num_classes + j];
                    float score = 1.0f / (1.0f + exp(-logit_value)); // Sigmoid

                    if (score > PRED_THRESHOLD && j < static_cast<int64_t>(text_queries_.size())) {
                        float x_center = pred_boxes[i * 4 + 0];
                        float y_center = pred_boxes[i * 4 + 1];
                        float width = pred_boxes[i * 4 + 2];
                        float height = pred_boxes[i * 4 + 3];

                        float x1 = (x_center - width / 2.0f) * orig_w;
                        float y1 = (y_center - height / 2.0f) * orig_h;
                        float x2 = (x_center + width / 2.0f) * orig_w;
                        float y2 = (y_center + height / 2.0f) * orig_h;

                        cv::rectangle(img, cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                                      cv::Point(static_cast<int>(x2), static_cast<int>(y2)),
                                      cv::Scalar(0, 255, 0), 2);
                        cv::putText(img, text_queries_[j], cv::Point(static_cast<int>(x1), static_cast<int>(y1) - 5),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

                        RCLCPP_INFO(this->get_logger(), "Detected '%s' with score %.2f at [%.2f, %.2f, %.2f, %.2f]",
                                    text_queries_[j].c_str(), score, x1, y1, x2, y2);
                    }
                }
            }

            // Save to the same images directory the node logs, and report the actual path used.
            const std::string out_path = "/home/bmacraze/ros2_kilted_ws/src/object_detection/images/output_image.jpg";
            cv::imwrite(out_path, img);
            RCLCPP_INFO(this->get_logger(), "Output image saved to: %s", out_path.c_str());
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