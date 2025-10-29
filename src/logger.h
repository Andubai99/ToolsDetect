#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>

#include "vision_pipeline.h"      // 旧的差分流程里用到 ToolBlob。如果你暂时不想用老方案，可以先保留这个 include，不会出错。
#include "inventory_compare.h"    // 新的 YOLO式库存差异结构

class Logger {
public:
    explicit Logger(const std::string& logPath)
        : logPath_(logPath)
    {}

    // 记录用户登录行为
    void logLogin(const std::string& username) {
        std::ostringstream oss;
        oss << timestamp()
            << " LOGIN user=\"" << username << "\"";
        appendLine(oss.str());
        std::cout << oss.str() << "\n";
    }

    // 旧差分法：记录每个检测到的变化区域（ToolBlob）
    void logToolEvent(const std::string& username,
                      const std::vector<ToolBlob>& blobs) {
        for (size_t i = 0; i < blobs.size(); ++i) {
            const auto& b = blobs[i];
            auto c = b.box.center;
            auto s = b.box.size;

            std::ostringstream oss;
            oss << timestamp()
                << " TOOL_CHANGE user=\"" << username << "\""
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
                << " TOOL_CHANGE user=\"" << username << "\""
                << " no_change";
            appendLine(oss.str());
            std::cout << oss.str() << "\n";
        }
    }

    // ✅ 新增：YOLO式库存差异日志
    // 把 compareInventory() 得到的 InventoryDelta 结果落库
    // 例子：
    // [2025-10-26 15:35:12] INVENTORY user="alice" pliers=-1 screwdriver=0 wrench=+1
    void logInventoryDelta(const std::string& username,
                           const InventoryDelta& delta) {
        std::ostringstream oss;
        oss << timestamp()
            << " INVENTORY user=\"" << username << "\"";

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

    // 生成类似 [2025-10-26 15:32:07]
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

    // 统一写入文件和追加换行
    void appendLine(const std::string& line) const {
        std::ofstream fout(logPath_, std::ios::app);
        if (!fout.is_open()) {
            std::cerr << "[ERROR] Cannot open log file: " << logPath_ << "\n";
            return;
        }
        fout << line << "\n";
    }
};
