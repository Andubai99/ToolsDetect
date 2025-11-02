#pragma once
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include "inventory_compare.h"

// 代表本轮检测产生的报警信息
struct AlarmInfo {
    bool triggered = false;              // 是否有告警
    std::vector<std::string> messages;   // 告警原因的文本描述（可能不止一个）
};

// 根据库存差异生成报警信息：
// 规则：
// 1. 任何 diff < 0  => 工具被拿走 -> 报警
// 2. 任何 diff > 0  => 柜内出现新增物品(可能是外来物) -> 报警
inline AlarmInfo evaluateAlarm(const InventoryDelta& delta) {
    AlarmInfo info;

    for (const auto& kv : delta.classCountDiff) {
        const std::string& cls = kv.first;
        int diff = kv.second;

        if (diff < 0) {
            info.triggered = true;
            // 例：pliers 变少1
            info.messages.push_back(
                "MISSING " + cls + " (" + std::to_string(-diff) + " removed)"
            );
        } else if (diff > 0) {
            info.triggered = true;
            // 例：多出一个未知/外来物
            info.messages.push_back(
                "ADDED " + cls + " (" + std::to_string(diff) + " new)"
            );
        }
    }

    return info;
}

// 实际触发报警（当前阶段 = 控制台输出+高亮标志）
// 将来可以升级为蜂鸣器/弹窗/语音播报
inline void raiseAlarmToConsole(const AlarmInfo& info,
                                const std::string& sessionId,
                                const std::string& username)
{
    if (!info.triggered) {
        std::cout << "[ALARM] No alarm for session " << sessionId
                  << " (user=" << username << ")\n";
        return;
    }

    std::cout << "==================== ALERT ====================\n";
    std::cout << "[ALARM] Session " << sessionId
              << " User=" << username << "\n";

    for (const auto& msg : info.messages) {
        std::cout << "  * " << msg << "\n";
    }

    std::cout << "================================================\n";
}
