#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <limits>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>

#include "auth.h"                // AuthManager
#include "logger.h"              // Logger (now logs alarm info)
#include "detector.h"            // runYoloDetect()
#include "inventory_compare.h"   // compareInventory()
#include "alert.h"               // evaluateAlarm(), raiseAlarmToConsole()

// 抽象的图像采集接口（第2周引入、第6周会接相机）
static cv::Mat captureBefore() {
    return cv::imread("testimg.jpg");
}
static cv::Mat captureAfter() {
    return cv::imread("testimg.jpg");
}

// 获取 "YYYY-MM-DD"
static std::string getCurrentDayString() {
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

int main() {
    std::cout << "=== ToolsDetect System (Week 3 baseline with ALARM) ===\n";

    // ------------------ 登录 ------------------
    AuthManager auth;
    if (!auth.loadUsers("users.txt")) {
        std::cerr << "[FATAL] Failed to load users.txt. Exiting.\n";
        std::cout << "Press ENTER to exit.\n";
        std::cin.get();
        return 1;
    }

    std::string username;
    std::string password;

    std::cout << "Username: ";
    std::cin >> username;
    std::cout << "Password: ";
    std::cin >> password;

    if (!auth.authenticate(username, password)) {
        std::cerr << "[ERROR] Authentication failed for user '" << username << "'.\n";
        std::cout << "Press ENTER to exit.\n";
        std::cin.ignore();
        std::cin.get();
        return 1;
    }

    std::cout << "[INFO] Login success. Welcome, " << username << "!\n";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // 清掉缓冲

    Logger logger("log.txt");
    logger.logLogin(username);

    // ---- 单次示例：对项目目录下的 testimg.jpg 运行 ONNX 推理 ----
    {
        const std::string testImagePath = "testimg.jpg";
        cv::Mat testImage = cv::imread(testImagePath);
        if (testImage.empty()) {
            std::cerr << "[WARN] Failed to load " << testImagePath
                      << ". Skip single-image inference preview.\n";
        } else {
            std::cout << "[INFO] Running YOLO ONNX inference on " << testImagePath << "...\n";
            DetectionResult previewResult = runYoloDetect(testImage);
            if (previewResult.objects.empty()) {
                std::cout << "[INFO] No detections found in " << testImagePath << ".\n";
            } else {
                for (const auto& obj : previewResult.objects) {
                    std::cout << "  - " << obj.cls
                              << " conf=" << obj.confidence
                              << " bbox=(" << obj.bbox.x << "," << obj.bbox.y
                              << "," << obj.bbox.width << "," << obj.bbox.height << ")\n";
                }
            }
        }
    }

    // sessionId 按天计数：YYYY-MM-DD#N
    std::string currentDay = getCurrentDayString();
    int dailyCounter = 0;

    // ------------------ 循环，一轮=一次开柜检查 ------------------
    while (true) {
        std::cout << "\n--- New inventory check session ---\n";

        // 计算当天第几次
        std::string today = getCurrentDayString();
        if (today != currentDay) {
            currentDay = today;
            dailyCounter = 1;
        } else {
            dailyCounter += 1;
        }
        std::string sessionId = currentDay + "#" + std::to_string(dailyCounter);

        // 性能计时开始
        auto t_start = std::chrono::high_resolution_clock::now();

        // 采集图像
        cv::Mat img_before = captureBefore();
        cv::Mat img_after  = captureAfter();

        if (img_before.empty() || img_after.empty()) {
            std::cerr << "[ERROR] Can't load before/after images.\n";
            std::cerr << "Ensure test_before.png / test_after.png are next to the exe.\n";
            break;
        }

        // 模型检测（当前还是占位，用runYoloDetect）
        DetectionResult det_before = runYoloDetect(img_before);
        DetectionResult det_after  = runYoloDetect(img_after);

        // 盘点差异
        InventoryDelta delta = compareInventory(det_before, det_after);

        // 告警判断
        AlarmInfo alarmInfo = evaluateAlarm(delta);

        // 如果有告警，立刻“喊出来”
        raiseAlarmToConsole(alarmInfo, sessionId, username);

        // 打印差异到控制台
        std::cout << "[INFO] Delta (after - before):\n";
        for (const auto& kv : delta.classCountDiff) {
            std::cout << "  " << kv.first << " -> " << kv.second << "\n";
        }

        // 性能计时结束
        auto t_end = std::chrono::high_resolution_clock::now();
        auto durationMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

        std::cout << "[PERF] Session " << sessionId
                  << " took " << durationMs << " ms\n";

        // 写日志（包含告警信息、sessionId、耗时、责任人）
        logger.logInventoryDelta(username, delta, sessionId, durationMs, alarmInfo);

        // 可视化 after（检测框）
        cv::Mat vis_after = img_after.clone();
        for (const auto& obj : det_after.objects) {
            cv::rectangle(vis_after, obj.bbox, cv::Scalar(0,255,0), 2);
            cv::putText(
                vis_after,
                obj.cls + " " + std::to_string(obj.confidence),
                cv::Point(obj.bbox.x, obj.bbox.y - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                2.0,
                cv::Scalar(0,255,0),
                4
            );
        }

        // 如果有告警，可以在窗口标题上也打标记
        std::string afterWindowTitle = "After Snapshot + Detections";
        if (alarmInfo.triggered) {
            afterWindowTitle += " [ALARM!]";
        }

        const int windowWidth = 1920;
        const int windowHeight = 1080;

        cv::namedWindow("Before Snapshot", cv::WINDOW_NORMAL);
        cv::resizeWindow("Before Snapshot", windowWidth, windowHeight);
        cv::imshow("Before Snapshot", img_before);

        cv::namedWindow(afterWindowTitle, cv::WINDOW_NORMAL);
        cv::resizeWindow(afterWindowTitle, windowWidth, windowHeight);
        cv::imshow(afterWindowTitle, vis_after);

        std::cout << "Press any key in the image window to continue...\n";
        cv::waitKey(0);

        // 下一轮 or 退出
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

    std::cout << "[INFO] System shutdown.\n";
    return 0;
}
