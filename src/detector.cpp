#include "detector.h"

#include "yoloinfer.h"

#include <exception>
#include <iostream>
#include <memory>
#include <mutex>

namespace {

YoloInfer* getSharedInfer() {
    static std::unique_ptr<YoloInfer> infer;
    static std::once_flag init_flag;

    std::call_once(init_flag, [&]() {
        try {
            infer = std::make_unique<YoloInfer>();
        } catch (const std::exception& ex) {
            std::cerr << "[ERROR] Failed to initialize YoloInfer: "
                      << ex.what() << "\n";
        }
    });

    return infer.get();
}

}  // namespace

DetectionResult runYoloDetect(const cv::Mat& img) {
    DetectionResult result;
    if (img.empty()) {
        std::cerr << "[WARN] runYoloDetect got empty image.\n";
        return result;
    }

    YoloInfer* infer = getSharedInfer();
    if (!infer) {
        std::cerr << "[ERROR] YoloInfer unavailable. Check ONNX configuration.\n";
        return result;
    }

    auto detections = infer->infer(img);
    for (const auto& det : detections) {
        DetectedObject obj;
        obj.cls = infer->classNameOrDefault(det.class_id);
        obj.confidence = det.score;
        obj.bbox = det.box;
        result.objects.push_back(obj);
    }

    return result;
}
