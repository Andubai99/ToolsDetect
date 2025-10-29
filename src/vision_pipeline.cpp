#include "vision_pipeline.h"
#include <iostream>
#include <cstdio>

// 小工具：安全减法（避免负值截断问题）
static cv::Mat safeSubtract(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat out;
    cv::subtract(a, b, out);  // 对应文中“差异化运算”：cv2.subtract(str1, str2):contentReference[oaicite:2]{index=2}
    return out;
}

// 图像加权融合：把两次差分结果的信息合到一起
// 文中描述用加权和 + 通道相加，得到“包含更多原图信息的图片”，再转灰度:contentReference[oaicite:3]{index=3}
static cv::Mat fuseDiffsToGray(const cv::Mat& d1, const cv::Mat& d2) {
    cv::Mat blended;
    // 0.5, 0.5 只是示例权重，对应文章里的 addWeighted() 思想:contentReference[oaicite:4]{index=4}
    cv::addWeighted(d1, 0.5, d2, 0.5, 0.0, blended);

    cv::Mat gray;
    if (blended.channels() == 3) {
        // 转单通道灰度，相当于把RGB通道信息压到一起（文中将各通道相加得到单通道）:contentReference[oaicite:5]{index=5}
        cv::cvtColor(blended, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = blended.clone();
    }
    return gray;
}

// 自适应阈值（二值化）：文章用大津法 OTSU 做阈值分割，得到黑白前景-背景分离:contentReference[oaicite:6]{index=6}
static cv::Mat otsuThreshold(const cv::Mat& gray) {
    cv::Mat bin;
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return bin;
}

// 形态学闭运算 = 膨胀后腐蚀，用来填孔、连碎块、去小噪声
// 文章中强调“先膨胀后腐蚀的闭运算…填补孔洞、消除小噪声、让轮廓更连续”:contentReference[oaicite:7]{index=7}
static cv::Mat morphClose(const cv::Mat& bin, int ksize = 5) {
    cv::Mat closed;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize, ksize));
    cv::morphologyEx(bin, closed, cv::MORPH_CLOSE, kernel);
    return closed;
}

// 从二值图提取轮廓，测量每个目标的最小外接矩形
// 对应文章的“cv2.findContours() + cv2.minAreaRect() -> 中心坐标、宽高、角度”:contentReference[oaicite:8]{index=8}
static std::vector<ToolBlob> extractBlobs(const cv::Mat& bin, double minArea = 200.0) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<ToolBlob> blobs;
    blobs.reserve(contours.size());

    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < minArea)
            continue; // 面积阈值过滤小噪声，文章中强调“面积阈值筛选”:contentReference[oaicite:9]{index=9}

        cv::RotatedRect box = cv::minAreaRect(c); // 最小外接矩形：中心(x,y)、宽高、旋转角度:contentReference[oaicite:10]{index=10}

        ToolBlob blob;
        blob.box = box;
        blob.area = area;
        blob.contour = c;
        blobs.push_back(blob);
    }

    return blobs;
}

std::vector<ToolBlob> detectToolChanges(
    const cv::Mat& beforeImg,
    const cv::Mat& afterImg,
    cv::Mat& debugBinary,
    cv::Mat& debugVis,
    const std::string& debugPrefix
) {
    if (beforeImg.empty() || afterImg.empty()) {
        std::cerr << "[ERROR] Input images are empty.\n";
        return {};
    }

    // Step 1: 差异化运算（两次减法，获取取放前后变化区域）:contentReference[oaicite:11]{index=11}
    // diff1 = after - before
    // diff2 = before - after
    cv::Mat diff1 = safeSubtract(afterImg, beforeImg);
    cv::Mat diff2 = safeSubtract(beforeImg, afterImg);

    // Step 2: 融合两次差分结果，得到更完整的变化区域信息:contentReference[oaicite:12]{index=12}
    cv::Mat fusedGray = fuseDiffsToGray(diff1, diff2);

    // Step 3: OTSU 二值化，提取显著变化区域（工具新增/缺失的地方）:contentReference[oaicite:13]{index=13}
    cv::Mat bin = otsuThreshold(fusedGray);

    // Step 4: 闭运算（膨胀+腐蚀）平滑区域、连通破碎轮廓，去掉小孔洞:contentReference[oaicite:14]{index=14}
    cv::Mat clean = morphClose(bin, 5);

    // Step 5: 连通域 / 轮廓提取 + 面积过滤 + 最小外接矩形测量:contentReference[oaicite:15]{index=15}
    std::vector<ToolBlob> blobs = extractBlobs(clean, /*minArea=*/200.0);

    // debug 可视化：在 after 图上画框和中心
    debugVis = afterImg.clone();
    drawToolDetections(debugVis, blobs);

    debugBinary = clean;

    // 可选：保存中间图，方便肉眼确认
    if (!debugPrefix.empty()) {
        cv::imwrite(debugPrefix + "_diff1.png", diff1);
        cv::imwrite(debugPrefix + "_diff2.png", diff2);
        cv::imwrite(debugPrefix + "_fusedGray.png", fusedGray);
        cv::imwrite(debugPrefix + "_bin.png", bin);
        cv::imwrite(debugPrefix + "_clean.png", clean);
        cv::imwrite(debugPrefix + "_vis.png", debugVis);
    }

    return blobs;
}

void drawToolDetections(
    cv::Mat& canvas,
    const std::vector<ToolBlob>& blobs
) {
    for (size_t i = 0; i < blobs.size(); ++i) {
        const auto& b = blobs[i];

        // 画最小外接矩形
        cv::Point2f pts[4];
        b.box.points(pts);
        for (int j = 0; j < 4; ++j) {
            cv::line(canvas, pts[j], pts[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }

        // 画中心点
        cv::circle(canvas, b.box.center, 4, cv::Scalar(0, 0, 255), cv::FILLED);

        // 标注信息：中心坐标、宽×高、角度
        char label[256];
        cv::Size2f sz = b.box.size;
        std::snprintf(
            label, sizeof(label),
            "#%zu (cx=%.1f, cy=%.1f) w=%.1f h=%.1f ang=%.1f area=%.1f",
            i,
            b.box.center.x, b.box.center.y,
            sz.width, sz.height,
            b.box.angle,
            b.area
        );

        cv::putText(
            canvas,
            label,
            b.box.center + cv::Point2f(5, -5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 255),
            1
        );
    }
}
