#include "auth.h"
#include <fstream>
#include <sstream>
#include <iostream>

bool AuthManager::loadUsers(const std::string& path) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "[ERROR] Cannot open user file: " << path << "\n";
        return false;
    }

    userPass_.clear();

    std::string line;
    int lineNo = 0;
    while (std::getline(fin, line)) {
        ++lineNo;
        if (line.empty()) continue;
        if (line[0] == '#') continue; // 允许注释行

        std::istringstream iss(line);
        std::string u, p;
        if (!(iss >> u >> p)) {
            std::cerr << "[WARN] Bad format in " << path << " line " << lineNo << "\n";
            continue;
        }
        userPass_[u] = p;
    }

    std::cout << "[INFO] Loaded " << userPass_.size() << " user(s) from " << path << "\n";
    return true;
}

bool AuthManager::authenticate(const std::string& user, const std::string& pass) const {
    auto it = userPass_.find(user);
    if (it == userPass_.end()) return false;
    return (it->second == pass);
}
