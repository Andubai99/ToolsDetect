#include "inventory_compare.h"

#include <unordered_map>

static std::unordered_map<std::string, int> countByClass(
    const DetectionResult& det
) {
    std::unordered_map<std::string, int> counts;
    for (const auto& obj : det.objects) {
        counts[obj.cls] += 1;
    }
    return counts;
}

InventoryDelta compareInventory(
    const DetectionResult& beforeDet,
    const DetectionResult& afterDet
) {
    InventoryDelta delta;
    delta.beforeDet = beforeDet;
    delta.afterDet  = afterDet;

    auto beforeMap = countByClass(beforeDet);
    auto afterMap  = countByClass(afterDet);

    // union of keys
    std::unordered_map<std::string, bool> allKeys;
    for (auto& kv : beforeMap) allKeys[kv.first] = true;
    for (auto& kv : afterMap)  allKeys[kv.first] = true;

    for (auto& kv : allKeys) {
        const std::string& cls = kv.first;
        int b = beforeMap.count(cls) ? beforeMap[cls] : 0;
        int a = afterMap.count(cls)  ? afterMap[cls]  : 0;
        delta.classCountDiff[cls] = a - b;
    }

    return delta;
}
