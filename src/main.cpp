#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <limits>

#include "auth.h"                // 你已有的认证模块 (AuthManager or Auth)
#include "logger.h"              // 刚刚更新的 Logger
#include "detector.h"            // YOLO风格检测接口（占位实现目前返回假数据）
#include "inventory_compare.h"   // compareInventory() -> InventoryDelta

int main() {
    std::cout << "=== ToolsDetect System (Week 1 baseline) ===\n";

    // 1. 登录
    AuthManager auth; // 如果你这边的类名是 Auth 而不是 AuthManager，就改成 Auth
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
    // Flush trailing newline so that the session loop can accept an empty ENTER.
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // 2. 初始化日志器
    Logger logger("log.txt");
    logger.logLogin(username);

    // 3. 多轮检测循环
    while (true) {
        std::cout << "\n--- New inventory check session ---\n";

        // 3.1 读取两张图（后面第6周我们会把它替换成摄像头抓拍）
        cv::Mat img_before = cv::imread("test_before.png");
        cv::Mat img_after  = cv::imread("test_after.png");

        if (img_before.empty() || img_after.empty()) {
            std::cerr << "[ERROR] Can't load test_before.png or test_after.png.\n";
            std::cerr << "Put them next to the exe.\n";
            break;
        }

        // 3.2 YOLO式检测（当前是占位版）
        DetectionResult det_before = runYoloDetect(img_before);
        DetectionResult det_after  = runYoloDetect(img_after);

        // 3.3 盘点差异
        InventoryDelta delta = compareInventory(det_before, det_after);

        // 3.4 控制台输出本轮差异
        std::cout << "[INFO] Delta (after - before):\n";
        for (const auto& kv : delta.classCountDiff) {
            std::cout << "  " << kv.first << " -> " << kv.second << "\n";
        }

        // 3.5 写日志（核心留痕点）
        logger.logInventoryDelta(username, delta);

        // 3.6 简单可视化：把 after 的检测框画出来展示
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

        // 3.7 询问是否继续
        std::cout << "Press ENTER for next round, or type q then ENTER to quit: "; // 按回车 = 模拟一次柜门关闭触发检测
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
