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
// Runs a video detection loop on the provided path. Returns true if the video
// stream was opened and processed, false otherwise (e.g., invalid path).
bool runVideoDetection(const std::string& videoPath);
