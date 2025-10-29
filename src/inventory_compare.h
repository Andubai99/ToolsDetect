#pragma once
#include "detector.h"
#include <map>
#include <string>

struct InventoryDelta {
    // 类别 -> 数量变化 (afterCount - beforeCount)
    // 例如: "pliers": -1 意味着钳子少了一把
    //       "screwdriver": +1 意味着多出一把螺丝刀
    std::map<std::string, int> classCountDiff;

    // 原始检测结果也保留，便于后续记录位置信息等
    DetectionResult beforeDet;
    DetectionResult afterDet;
};

// 根据两次检测结果（开柜前 & 关柜后）
// 计算每个类别的数量变化
InventoryDelta compareInventory(
    const DetectionResult& beforeDet,
    const DetectionResult& afterDet
);
