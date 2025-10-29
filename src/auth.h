#pragma once
#include <string>
#include <unordered_map>

// 负责加载用户表，和校验用户名/密码
class AuthManager {
public:
    // 从一个简单的文本文件加载用户
    // 文件格式示例（每行一个用户）:
    //   alice password123
    //   bob qwerty
    //
    // 返回 true 表示加载成功
    bool loadUsers(const std::string& path);

    // 校验用户名和密码
    bool authenticate(const std::string& user, const std::string& pass) const;

private:
    std::unordered_map<std::string, std::string> userPass_;
};
