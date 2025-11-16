#include "session_runner.h"

#include "alert.h"
#include "detector.h"
#include "inventory_compare.h"
#include "logger.h"
#include "opencv_config.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <cerrno>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace {

cv::Mat captureBefore() {
    return cv::imread("t1.jpg");
}

cv::Mat captureAfter() {
    return cv::imread("t2.jpg");
}

struct DrawOverlayStyle {
    cv::Scalar color{0, 255, 0};
    int rectThickness = 2;
    double fontScale = 1.0;
    int textThickness = 2;
};

DrawOverlayStyle makeOverlayStyle(double relativeScale) {
    DrawOverlayStyle style;
    style.rectThickness =
        std::max(1, static_cast<int>(std::round(4.0 * relativeScale)));
    style.fontScale = 2.5 * relativeScale;
    style.textThickness =
        std::max(1, static_cast<int>(std::round(5.0 * relativeScale)));
    return style;
}

double computeImageDiagonal(const cv::Mat& image) {
    if (image.empty()) {
        return 0.0;
    }
    return std::hypot(static_cast<double>(image.cols),
                      static_cast<double>(image.rows));
}

void drawDetections(cv::Mat& image,
                    const DetectionResult& detections,
                    const DrawOverlayStyle& style) {
    for (const auto& obj : detections.objects) {
        cv::rectangle(image, obj.bbox, style.color, style.rectThickness);
        cv::putText(
            image,
            obj.cls + " " + std::to_string(obj.confidence),
            cv::Point(obj.bbox.x, obj.bbox.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            style.fontScale,
            style.color,
            style.textThickness
        );
    }
}

std::string getCurrentDayString() {
    auto now = std::chrono::system_clock::now();
    std::time_t t  = std::chrono::system_clock::to_time_t(now);
    std::tm      tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d");
    return oss.str();
}

void runSingleImagePreview() {
    const std::string testImagePath = "t1.jpg";
    cv::Mat testImage = cv::imread(testImagePath);
    if (testImage.empty()) {
        std::cerr << "[WARN] Failed to load " << testImagePath
                  << ". Skip single-image inference preview.\n";
        return;
    }

    std::cout << "[INFO] Running YOLO ONNX inference on " << testImagePath
              << "...\n";
    DetectionResult previewResult = runYoloDetect(testImage);
    if (previewResult.objects.empty()) {
        std::cout << "[INFO] No detections found in " << testImagePath << ".\n";
        return;
    }

    for (const auto& obj : previewResult.objects) {
        std::cout << "  - " << obj.cls
                  << " conf=" << obj.confidence
                  << " bbox=(" << obj.bbox.x << "," << obj.bbox.y
                  << "," << obj.bbox.width << "," << obj.bbox.height << ")\n";
    }
}

}  // namespace

bool ensureDirectoryExists(const std::string& dir) {
#if defined(_WIN32)
    int ret = _mkdir(dir.c_str());
#else
    int ret = mkdir(dir.c_str(), 0755);
#endif
    if (ret == 0 || errno == EEXIST) {
        return true;
    }
    return false;
}

void runVideoDetection(const std::string& videoPath) {
#if TOOLSDETECT_HAS_VIDEOIO && TOOLSDETECT_HAS_HIGHGUI
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Failed to open video: " << videoPath << "\n";
        return;
    }

    std::cout << "[INFO] Starting video detection on " << videoPath
              << ". Press 'q' to exit video mode.\n";

    const std::string windowName = "Video Detection";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    const DrawOverlayStyle videoStyle = makeOverlayStyle(1.0);

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cout << "[INFO] Video stream ended.\n";
            break;
        }

        DetectionResult detections = runYoloDetect(frame);
        cv::Mat vis = frame.clone();
        drawDetections(vis, detections, videoStyle);

        cv::imshow(windowName, vis);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            std::cout << "[INFO] Video detection interrupted by user.\n";
            break;
        }
    }

    cv::destroyWindow(windowName);
#else
    (void)videoPath;
    std::cerr << "[ERROR] Video mode is unavailable "
                 "(OpenCV videoio/highgui not found at build time).\n";
#endif
}

void runBeforeAfterSessions(Logger& logger,
                            const std::string& username,
                            const std::string& resultsDir) {
    runSingleImagePreview();

    std::string currentDay = getCurrentDayString();
    int dailyCounter = 0;

    while (true) {
        std::cout << "\n--- New inventory check session ---\n";

        std::string today = getCurrentDayString();
        if (today != currentDay) {
            currentDay = today;
            dailyCounter = 1;
        } else {
            dailyCounter += 1;
        }
        std::string sessionId = currentDay + "#" + std::to_string(dailyCounter);

        auto t_start = std::chrono::high_resolution_clock::now();

        cv::Mat img_before = captureBefore();
        cv::Mat img_after  = captureAfter();

        if (img_before.empty() || img_after.empty()) {
            std::cerr << "[ERROR] Can't load before/after images.\n";
            std::cerr << "Ensure test_before.jpg / test_after.jpg are next to the exe.\n";
            break;
        }

        DetectionResult det_before = runYoloDetect(img_before);
        DetectionResult det_after  = runYoloDetect(img_after);

        InventoryDelta delta = compareInventory(det_before, det_after);
        AlarmInfo alarmInfo = evaluateAlarm(delta);
        raiseAlarmToConsole(alarmInfo, sessionId, username);

        std::cout << "[INFO] Delta (after - before):\n";
        for (const auto& kv : delta.classCountDiff) {
            std::cout << "  " << kv.first << " -> " << kv.second << "\n";
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        auto durationMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

        std::cout << "[PERF] Session " << sessionId
                  << " took " << durationMs << " ms\n";

        logger.logInventoryDelta(username, delta, sessionId, durationMs, alarmInfo);

        const double beforeDiag = computeImageDiagonal(img_before);
        const double afterDiag = computeImageDiagonal(img_after);
        const double relativeAfterScale =
            (beforeDiag > 0.0) ? (afterDiag / beforeDiag) : 1.0;
        const DrawOverlayStyle beforeStyle = makeOverlayStyle(1.0);
        const DrawOverlayStyle afterStyle = makeOverlayStyle(relativeAfterScale);

        cv::Mat vis_before = img_before.clone();
        cv::Mat vis_after = img_after.clone();
        drawDetections(vis_before, det_before, beforeStyle);
        drawDetections(vis_after, det_after, afterStyle);

        const std::string beforeResultPath = resultsDir + "/" + sessionId + "_before.jpg";
        const std::string afterResultPath = resultsDir + "/" + sessionId + "_after.jpg";
        if (cv::imwrite(beforeResultPath, vis_before)) {
            std::cout << "[INFO] Saved detection visualization to "
                      << beforeResultPath << "\n";
        } else {
            std::cerr << "[ERROR] Failed to write detection visualization to "
                      << beforeResultPath << "\n";
        }
        if (cv::imwrite(afterResultPath, vis_after)) {
            std::cout << "[INFO] Saved detection visualization to "
                      << afterResultPath << "\n";
        } else {
            std::cerr << "[ERROR] Failed to write detection visualization to "
                      << afterResultPath << "\n";
        }

        std::string afterWindowTitle = "After Snapshot + Detections";
        if (alarmInfo.triggered) {
            afterWindowTitle += " [ALARM!]";
        }

        const int windowWidth = 1920;
        const int windowHeight = 1080;

        cv::namedWindow("Before Snapshot", cv::WINDOW_NORMAL);
        cv::resizeWindow("Before Snapshot", windowWidth, windowHeight);
        cv::imshow("Before Snapshot", vis_before);

        cv::namedWindow(afterWindowTitle, cv::WINDOW_NORMAL);
        cv::resizeWindow(afterWindowTitle, windowWidth, windowHeight);
        cv::imshow(afterWindowTitle, vis_after);

        std::cout << "Press any key in the image window to continue...\n";
        cv::waitKey(0);

        std::cout << "Press ENTER for next round, or type q then ENTER to quit: ";
        std::string cmd;
        if (!std::getline(std::cin, cmd)) {
            std::cerr << "[WARN] Input stream closed. Exiting session loop.\n";
            break;
        }
        if (cmd == "q" || cmd == "Q") {
            std::cout << "[INFO] Exiting session loop.\n";
            break;
        }
    }
}
