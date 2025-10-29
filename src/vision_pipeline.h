#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct ToolBlob {
    cv::RotatedRect box;   // 最小外接矩形（中心、尺寸、角度）
    double area;           // 轮廓面积
    std::vector<cv::Point> contour; // 原始轮廓
};

// 图像差异 + 预处理 + 提取工具区域
// beforeImg: 取/放之前的柜内图
// afterImg:  取/放之后的柜内图
// debugPrefix: 用于保存/显示中间结果时的前缀（比如 "debug_"）
std::vector<ToolBlob> detectToolChanges(
    const cv::Mat& beforeImg,
    const cv::Mat& afterImg,
    cv::Mat& debugBinary,
    cv::Mat& debugVis,
    const std::string& debugPrefix = ""
);

// 画检测结果（标注中心点、角度、宽高等信息）
void drawToolDetections(
    cv::Mat& canvas,
    const std::vector<ToolBlob>& blobs
);
