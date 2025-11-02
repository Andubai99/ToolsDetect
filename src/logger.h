#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>

#include "vision_pipeline.h"
#include "inventory_compare.h"
#include "alert.h" // <-- 新增

class Logger {
public:
    explicit Logger(const std::string& logPath)
        : logPath_(logPath)
    {}

    void logLogin(const std::string& username) {
        std::ostringstream oss;
        oss << timestamp()
            << " LOGIN"
            << " user=\"" << username << "\"";
        appendLine(oss.str());
        std::cout << oss.str() << "\n";
    }

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

    // 第三周增强版：
    //  - 记录报警状态 alarm="YES"/"NO"
    //  - 若YES, 附上报警原因串
    //
    void logInventoryDelta(const std::string& username,
                           const InventoryDelta& delta,
                           const std::string& sessionId,
                           long long durationMs,
                           const AlarmInfo& alarmInfo)
    {
        std::ostringstream oss;
        oss << timestamp()
            << " INVENTORY"
            << " user=\"" << username << "\""
            << " session=\"" << sessionId << "\""
            << " duration_ms=" << durationMs
            << " alarm=" << (alarmInfo.triggered ? "YES" : "NO");

        // 报警详细原因
        if (alarmInfo.triggered && !alarmInfo.messages.empty()) {
            oss << " alarm_reasons=\"";
            for (size_t i = 0; i < alarmInfo.messages.size(); ++i) {
                if (i > 0) oss << "; ";
                oss << alarmInfo.messages[i];
            }
            oss << "\"";
        }

        // 差异明细
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

    void appendLine(const std::string& line) const {
        std::ofstream fout(logPath_, std::ios::app);
        if (!fout.is_open()) {
            std::cerr << "[ERROR] Cannot open log file: " << logPath_ << "\n";
            return;
        }
        fout << line << "\n";
    }
};
