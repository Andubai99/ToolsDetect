#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>

#include "vision_pipeline.h"      // 旧的差分法还保留着，没坏处
#include "inventory_compare.h"    // InventoryDelta 定义，含 classCountDiff

class Logger {
public:
    explicit Logger(const std::string& logPath)
        : logPath_(logPath)
    {}

    // 登录记录
    void logLogin(const std::string& username) {
        std::ostringstream oss;
        oss << timestamp()
            << " LOGIN"
            << " user=\"" << username << "\"";
        appendLine(oss.str());
        std::cout << oss.str() << "\n";
    }

    // 旧方案：差分出来的目标区域（保留兼容性，后续可能不再用）
    void logToolEvent(const std::string& username,
                      const std::vector<ToolBlob>& blobs) {
        for (size_t i = 0; i < blobs.size(); ++i) {
            const auto& b = blobs[i];
            auto c = b.box.center;
            auto s = b.box.size;

            std::ostringstream oss;
            oss << timestamp()
                << " TOOL_CHANGE"
                << " user=\"" << username << "\""
                << " id=" << i
                << " cx=" << c.x
                << " cy=" << c.y
                << " w=" << s.width
                << " h=" << s.height
                << " angle=" << b.box.angle
                << " area=" << b.area;
            appendLine(oss.str());
            std::cout << oss.str() << "\n";
        }

        if (blobs.empty()) {
            std::ostringstream oss;
            oss << timestamp()
                << " TOOL_CHANGE"
                << " user=\"" << username << "\""
                << " no_change";
            appendLine(oss.str());
            std::cout << oss.str() << "\n";
        }
    }

    // ✅ 第二周升级后的核心日志接口
    //
    // username: 当前操作用户
    // delta:    compareInventory() 的结果，包含每个类别的count差
    // sessionId: "YYYY-MM-DD#N" 例如 "2025-10-29#3" -> 表示2025-10-29当天第3次关柜检测
    // durationMs: 这一轮识别+比对+记录总耗时(毫秒)
    //
    // 日志示例：
    // [2025-10-29 14:03:55] INVENTORY user="alice" session="2025-10-29#3" duration_ms=1432 pliers=-1 screwdriver=0
    void logInventoryDelta(const std::string& username,
                           const InventoryDelta& delta,
                           const std::string& sessionId,
                           long long durationMs) {
        std::ostringstream oss;
        oss << timestamp()
            << " INVENTORY"
            << " user=\"" << username << "\""
            << " session=\"" << sessionId << "\""
            << " duration_ms=" << durationMs;

        for (const auto& kv : delta.classCountDiff) {
            const std::string& cls = kv.first;
            int diff = kv.second;
            oss << " " << cls << "=" << diff;
        }

        appendLine(oss.str());
        std::cout << oss.str() << "\n";
    }

private:
    std::string logPath_;

    // 时间戳，例如 [2025-10-29 14:03:55]
    std::string timestamp() const {
        auto now = std::chrono::system_clock::now();
        std::time_t t  = std::chrono::system_clock::to_time_t(now);
        std::tm      tm{};
    #if defined(_WIN32)
        localtime_s(&tm, &t);
    #else
        localtime_r(&t, &tm);
    #endif

        std::ostringstream oss;
        oss << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "]";
        return oss.str();
    }

    // 统一文件落盘
    void appendLine(const std::string& line) const {
        std::ofstream fout(logPath_, std::ios::app);
        if (!fout.is_open()) {
            std::cerr << "[ERROR] Cannot open log file: " << logPath_ << "\n";
            return;
        }
        fout << line << "\n";
    }
};
