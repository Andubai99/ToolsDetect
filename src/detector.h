#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// 一次检测到的单个目标（相当于 YOLO 的一条检测框）
struct DetectedObject {
    std::string cls;   // 类别，比如 "pliers", "screwdriver", "wrench"
    float confidence;  // 置信度 0~1
    cv::Rect bbox;     // 边界框 (x,y,w,h)
};

// 检测结果的打包
struct DetectionResult {
    std::vector<DetectedObject> objects;
};

// 这是占位接口：给一张图像，返回检测到的目标列表。
// 现在我们会用“假数据”来模拟输出，以后你把里面的实现替换成真正的 YOLO 推理即可。
DetectionResult runYoloDetect(const cv::Mat& img);
