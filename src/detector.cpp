#include "detector.h"
#include <iostream>

// 当前是占位实现：
// 实际项目里你会：
//   1. 预处理图像 (letterbox / resize / normalize).
//   2. 喂给 YOLO 推理引擎 (ONNX / TensorRT / ncnn / ...).
//   3. 解析输出，得到每个 (class, conf, bbox).
//
// 现在我们先返回假的数据，方便后续流程跑通。
// 你可以临时手动修改这里来模拟不同场景。

DetectionResult runYoloDetect(const cv::Mat& img) {
    DetectionResult result;

    if (img.empty()) {
        std::cerr << "[WARN] runYoloDetect got empty image.\n";
        return result;
    }

    // 示例：假装我们检测到了两种工具
    // 注意：这些只是演示，后续请替换成真实推理结果。

    // 假设看到一把钳子
    {
        DetectedObject obj;
        obj.cls = "pliers";     // 钳子
        obj.confidence = 0.92f;
        obj.bbox = cv::Rect(100, 200, 60, 30); // x,y,w,h
        result.objects.push_back(obj);
    }

    // 假设看到一把螺丝刀
    {
        DetectedObject obj;
        obj.cls = "screwdriver"; // 螺丝刀
        obj.confidence = 0.88f;
        obj.bbox = cv::Rect(300, 180, 80, 20);
        result.objects.push_back(obj);
    }

    return result;
}
