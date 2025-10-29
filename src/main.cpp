#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <limits>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>

#include "auth.h"                // AuthManager: loadUsers(), authenticate()
#include "logger.h"              // Logger with logInventoryDelta(session,durationMs)
#include "detector.h"            // runYoloDetect()
#include "inventory_compare.h"   // compareInventory()

// --- 摄像头采集占位层 (第2周先保持读取文件，第6周会改成真正拍照) ---
static cv::Mat captureBefore() {
    return cv::imread("test_before.png");
}
static cv::Mat captureAfter() {
    return cv::imread("test_after.png");
}

// --- 获取"YYYY-MM-DD"的日期字符串，用于session按天归档 ---
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
    std::cout << "=== ToolsDetect System (Week 2 baseline) ===\n";

    // 1. 登录 & 用户管理
    AuthManager auth;
    if (!auth.loadUsers("users.txt")) {  // 读取本地用户文件，支持离线运行 :contentReference[oaicite:4]{index=4}
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

    if (!auth.authenticate(username, password)) {  // 用户身份绑定，后续日志可追责 :contentReference[oaicite:5]{index=5}
        std::cerr << "[ERROR] Authentication failed for user '" << username << "'.\n";
        std::cout << "Press ENTER to exit.\n";
        std::cin.ignore();
        std::cin.get();
        return 1;
    }

    std::cout << "[INFO] Login success. Welcome, " << username << "!\n";
    // 清掉缓冲，避免后面 getline() 直接吃到换行
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // 2. 初始化日志器
    Logger logger("log.txt");
    logger.logLogin(username);

    // 用于生成“某年某月某日第x次开柜”
    std::string currentDay = getCurrentDayString();
    int dailyCounter = 0;

    // 3. 多轮检测循环（每轮 = 一次开柜→关柜事件）
    while (true) {
        std::cout << "\n--- New inventory check session ---\n";

        // ----- 会话计数：计算今天第几次 -----
        std::string today = getCurrentDayString();
        if (today != currentDay) {
            // 跨天了，从1重新计数
            currentDay = today;
            dailyCounter = 1;
        } else {
            // 同一天，+1
            dailyCounter += 1;
        }
        // 形如 "2025-10-29#3"
        std::string sessionId = currentDay + "#" + std::to_string(dailyCounter);

        // ----- 性能计时：一轮识别总耗时 -----
        auto t_start = std::chrono::high_resolution_clock::now();

        // 3.1 抓取 before / after 图像
        cv::Mat img_before = captureBefore();
        cv::Mat img_after  = captureAfter();

        if (img_before.empty() || img_after.empty()) {
            std::cerr << "[ERROR] Can't load before/after images.\n";
            std::cerr << "Ensure test_before.png / test_after.png are next to the exe.\n";
            break;
        }

        // 3.2 运行检测（当前是占位，后面会接真正的YOLO推理）
        DetectionResult det_before = runYoloDetect(img_before);
        DetectionResult det_after  = runYoloDetect(img_after);

        // 3.3 生成库存差异
        InventoryDelta delta = compareInventory(det_before, det_after);

        // 3.4 打印本轮差异到控制台
        std::cout << "[INFO] Delta (after - before):\n";
        for (const auto& kv : delta.classCountDiff) {
            std::cout << "  " << kv.first << " -> " << kv.second << "\n";
        }

        // ----- 性能计时结束 -----
        auto t_end = std::chrono::high_resolution_clock::now();
        auto durationMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

        std::cout << "[PERF] Session " << sessionId
                  << " took " << durationMs << " ms\n";

        // 3.5 记录日志（含 user / session 当天第几次 / 耗时 / 差异）
        logger.logInventoryDelta(username, delta, sessionId, durationMs);

        // 3.6 显示视觉结果（after图上叠检测框）
        cv::Mat vis_after = img_after.clone();
        for (const auto& obj : det_after.objects) {
            cv::rectangle(vis_after, obj.bbox, cv::Scalar(0,255,0), 2);
            cv::putText(
                vis_after,
                obj.cls + " " + std::to_string(obj.confidence),
                cv::Point(obj.bbox.x, obj.bbox.y - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(0,255,0),
                1
            );
        }

        cv::imshow("Before Snapshot", img_before);
        cv::imshow("After Snapshot + Detections", vis_after);
        std::cout << "Press any key in the image window to continue...\n";
        cv::waitKey(0);

        // 3.7 是否继续下一次柜门操作// 按回车 = 模拟一次柜门关闭触发检测
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
