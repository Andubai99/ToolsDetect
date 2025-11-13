// yoloinfer.h
// Bridges the ONNX Runtime based YOLO inference implementation so that other
// modules (detector/main) can trigger real model inference.

#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

// Default model/runtime configuration.
inline constexpr int kYoloInputWidth  = 640;
inline constexpr int kYoloInputHeight = 640;
inline constexpr float kYoloConfidenceThreshold = 0.25f;
inline constexpr float kYoloNmsThreshold        = 0.45f;
inline const std::wstring kDefaultModelPath =
    L"F:\\ultralytics-main\\ToolsDetect\\train35\\weights\\best.onnx";

struct YoloResult {
    int class_id = -1;
    float score = 0.0f;
    cv::Rect box;
};

// Returns the dataset class-name list (ToolsDetect).
std::vector<std::string> getDefaultToolClassNames();

class YoloInfer {
public:
    explicit YoloInfer(
        const std::wstring& model_path = kDefaultModelPath,
        int input_w = kYoloInputWidth,
        int input_h = kYoloInputHeight,
        float conf_thresh = kYoloConfidenceThreshold,
        float nms_thresh = kYoloNmsThreshold,
        std::vector<std::string> class_names = getDefaultToolClassNames()
    );

    // Run inference on an already-loaded image.
    std::vector<YoloResult> infer(const cv::Mat& image);

    // Convenience overload that reads from disk before inference.
    std::vector<YoloResult> infer(const std::string& image_path);

    const std::vector<std::string>& classNames() const { return class_names_; }
    std::string classNameOrDefault(int class_id) const;

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;

    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> output_shape_;

    int input_w_;
    int input_h_;
    float conf_thresh_;
    float nms_thresh_;
    std::vector<std::string> class_names_;
    std::wstring model_path_;
};
