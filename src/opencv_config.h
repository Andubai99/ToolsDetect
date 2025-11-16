#pragma once

#include <opencv2/opencv.hpp>

#if defined(__has_include)
#  if __has_include(<opencv2/videoio.hpp>)
#    include <opencv2/videoio.hpp>
#    define TOOLSDETECT_HAS_VIDEOIO 1
#  elif __has_include(<opencv2/videoio/videoio.hpp>)
#    include <opencv2/videoio/videoio.hpp>
#    define TOOLSDETECT_HAS_VIDEOIO 1
#  endif
#endif
#ifndef TOOLSDETECT_HAS_VIDEOIO
#define TOOLSDETECT_HAS_VIDEOIO 0
#endif

#if defined(__has_include)
#  if __has_include(<opencv2/highgui.hpp>)
#    include <opencv2/highgui.hpp>
#    define TOOLSDETECT_HAS_HIGHGUI 1
#  elif __has_include(<opencv2/highgui/highgui.hpp>)
#    include <opencv2/highgui/highgui.hpp>
#    define TOOLSDETECT_HAS_HIGHGUI 1
#  endif
#endif
#ifndef TOOLSDETECT_HAS_HIGHGUI
#define TOOLSDETECT_HAS_HIGHGUI 0
#endif
