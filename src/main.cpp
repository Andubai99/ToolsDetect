#include "opencv_config.h"

#include <iostream>
#include <string>

#include "auth.h"
#include "logger.h"
#include "session_runner.h"

int main() {
    std::cout << "=== ToolsDetect System (Week 3 baseline with ALARM) ===\n";

    AuthManager auth;
    if (!auth.loadUsers("users.txt")) {
        std::cerr << "[FATAL] Failed to load users.txt. Exiting.\n";
        std::cout << "Press ENTER to exit.\n";
        std::cin.get();
        return 1;
    }

    std::string username;
    std::string password;
    const int maxLoginAttempts = 3;
    bool authenticated = false;

    for (int attempt = 1; attempt <= maxLoginAttempts; ++attempt) {
        std::cout << "Username: ";
        if (!std::getline(std::cin, username)) {
            std::cerr << "[ERROR] Input stream closed while reading username.\n";
            return 1;
        }
        std::cout << "Password: ";
        if (!std::getline(std::cin, password)) {
            std::cerr << "[ERROR] Input stream closed while reading password.\n";
            return 1;
        }

        if (auth.authenticate(username, password)) {
            authenticated = true;
            break;
        }

        int remaining = maxLoginAttempts - attempt;
        std::cerr << "[ERROR] Authentication failed for user '" << username << "'.\n";
        if (remaining > 0) {
            std::cout << "[INFO] Please try again (" << remaining << " attempt(s) left).\n";
        }
    }

    if (!authenticated) {
        std::cout << "[FATAL] Maximum login attempts exceeded. Press ENTER to exit.\n";
        std::cin.get();
        return 1;
    }

    std::cout << "[INFO] Login success. Welcome, " << username << "!\n";

    Logger logger("log.txt");
    logger.logLogin(username);

    const std::string resultsDir = "results";
    if (!ensureDirectoryExists(resultsDir)) {
        std::cerr << "[WARN] Failed to create/access results directory.\n";
    }

    bool useVideoMode = false;
    while (true) {
        std::cout
            << "Select detection mode:\n"
            << "  [1] Before/after image sessions (default)\n"
            << "  [2] Video detection loop\n"
            << "Enter choice: ";
        std::string modeInput;
        if (!std::getline(std::cin, modeInput)) {
            std::cerr << "[ERROR] Input stream closed. Exiting.\n";
            return 1;
        }
        if (modeInput.empty() || modeInput == "1" || modeInput == "image" || modeInput == "Image") {
            useVideoMode = false;
            break;
        }
        if (modeInput == "2" || modeInput == "video" || modeInput == "Video") {
            useVideoMode = true;
            break;
        }
        std::cout << "[WARN] Invalid choice. Please enter 1 or 2.\n";
    }

#if !(TOOLSDETECT_HAS_VIDEOIO && TOOLSDETECT_HAS_HIGHGUI)
    if (useVideoMode) {
        std::cerr << "[ERROR] Video mode not supported in this build (missing OpenCV videoio/highgui modules).\n"
                  << "Install the required OpenCV components or rebuild with video support enabled.\n";
        return 1;
    }
#endif

    if (useVideoMode) {
        std::string videoPath;
        std::cout << "Enter video file path (e.g., data/video.mp4): ";
        if (!std::getline(std::cin, videoPath)) {
            std::cerr << "[ERROR] Input stream closed before video path. Exiting.\n";
            return 1;
        }
        if (videoPath.empty()) {
            videoPath = "video.mp4";
            std::cout << "[INFO] Using default video path: " << videoPath << "\n";
        }

        runVideoDetection(videoPath);
        std::cout << "[INFO] System shutdown.\n";
        return 0;
    }

    runBeforeAfterSessions(logger, username, resultsDir);

    std::cout << "[INFO] System shutdown.\n";
    return 0;
}
