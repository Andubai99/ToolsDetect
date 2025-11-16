#pragma once

#include <string>

class Logger;

// Ensures the specified directory exists (creates if necessary).
bool ensureDirectoryExists(const std::string& dir);

// Launches the before/after snapshot workflow (interactive loop).
void runBeforeAfterSessions(Logger& logger,
                            const std::string& username,
                            const std::string& resultsDir);

// Streams detections over a video file if the build has videoio/highgui.
void runVideoDetection(const std::string& videoPath);
