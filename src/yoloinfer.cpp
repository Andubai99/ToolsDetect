// yoloinfer.cpp
// ONNX Runtime + OpenCV YOLO inference helper.

#include "yoloinfer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

float iou(const cv::Rect& a, const cv::Rect& b) {
    int xx1 = std::max(a.x, b.x);
    int yy1 = std::max(a.y, b.y);
    int xx2 = std::min(a.x + a.width, b.x + b.width);
    int yy2 = std::min(a.y + a.height, b.y + b.height);
    int w = std::max(0, xx2 - xx1);
    int h = std::max(0, yy2 - yy1);
    int inter = w * h;
    int union_area = a.area() + b.area() - inter;
    if (union_area <= 0) return 0.0f;
    return static_cast<float>(inter) / static_cast<float>(union_area);
}

std::vector<int> nms(const std::vector<YoloResult>& dets, float iou_threshold) {
    std::vector<int> idxs;
    if (dets.empty()) return idxs;
    std::vector<int> order(dets.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = static_cast<int>(i);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return dets[a].score > dets[b].score;
    });

    std::vector<char> suppressed(dets.size(), 0);
    for (size_t oi = 0; oi < order.size(); ++oi) {
        int i = order[oi];
        if (suppressed[i]) continue;
        idxs.push_back(i);
        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            int j = order[oj];
            if (suppressed[j]) continue;
            if (iou(dets[i].box, dets[j].box) > iou_threshold) {
                suppressed[j] = 1;
            }
        }
    }
    return idxs;
}

void preprocess(const cv::Mat& img_bgr,
                std::vector<float>& out_tensor,
                int input_w,
                int input_h) {
    if (img_bgr.empty()) return;

    int orig_w = img_bgr.cols;
    int orig_h = img_bgr.rows;

    float r = std::min(static_cast<float>(input_w) / static_cast<float>(orig_w),
                       static_cast<float>(input_h) / static_cast<float>(orig_h));
    int new_unpad_w = static_cast<int>(std::round(orig_w * r));
    int new_unpad_h = static_cast<int>(std::round(orig_h * r));

    cv::Mat resized;
    cv::resize(img_bgr, resized, cv::Size(new_unpad_w, new_unpad_h));

    cv::Mat canvas(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    int dw = (input_w - new_unpad_w) / 2;
    int dh = (input_h - new_unpad_h) / 2;
    resized.copyTo(canvas(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

    cv::Mat img_rgb;
    cv::cvtColor(canvas, img_rgb, cv::COLOR_BGR2RGB);
    img_rgb.convertTo(img_rgb, CV_32F, 1.0f / 255.0f);

    std::vector<cv::Mat> channels(3);
    cv::split(img_rgb, channels);

    size_t img_size = static_cast<size_t>(input_w) * static_cast<size_t>(input_h);
    out_tensor.resize(3 * img_size);
    for (int c = 0; c < 3; ++c) {
        float* dst = out_tensor.data() + c * img_size;
        std::memcpy(dst, channels[c].data, img_size * sizeof(float));
    }
}

cv::Rect scale_coords_back(const cv::Rect& box,
                           int orig_w,
                           int orig_h,
                           int input_w,
                           int input_h) {
    float r = std::min(static_cast<float>(input_w) / static_cast<float>(orig_w),
                       static_cast<float>(input_h) / static_cast<float>(orig_h));
    int new_unpad_w = static_cast<int>(std::round(orig_w * r));
    int new_unpad_h = static_cast<int>(std::round(orig_h * r));
    int dw = (input_w - new_unpad_w) / 2;
    int dh = (input_h - new_unpad_h) / 2;

    float x = static_cast<float>(box.x);
    float y = static_cast<float>(box.y);
    float w = static_cast<float>(box.width);
    float h = static_cast<float>(box.height);

    float x_no_pad = (x - static_cast<float>(dw)) / r;
    float y_no_pad = (y - static_cast<float>(dh)) / r;
    float w_no_pad = w / r;
    float h_no_pad = h / r;

    int x0 = std::max(0, static_cast<int>(std::round(x_no_pad)));
    int y0 = std::max(0, static_cast<int>(std::round(y_no_pad)));
    int x1 = std::min(orig_w, static_cast<int>(std::round(x_no_pad + w_no_pad)));
    int y1 = std::min(orig_h, static_cast<int>(std::round(y_no_pad + h_no_pad)));
    return cv::Rect(x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0));
}

}  // namespace

std::vector<std::string> getDefaultToolClassNames() {
    return {
        "Adjustable Wrench",
        "Combination Wrench",
        "Double Box-end Wrench",
        "Double Open-end Wrench",
        "Hammer",
        "Level",
        "Mallet",
        "Nuts",
        "Pipe Wrench",
        "Pliers",
        "Pliers Wrench",
        "Screw",
        "Screwdriver",
        "Single Open-end Wrench",
        "Tape Measure",
        "washer"
    };
}

YoloInfer::YoloInfer(const std::wstring& model_path,
                     int input_w,
                     int input_h,
                     float conf_thresh,
                     float nms_thresh,
                     std::vector<std::string> class_names)
    : env_(ORT_LOGGING_LEVEL_WARNING, "yoloinfer"),
      session_(nullptr),
      allocator_(std::make_unique<Ort::AllocatorWithDefaultOptions>()),
      input_w_(input_w),
      input_h_(input_h),
      conf_thresh_(conf_thresh),
      nms_thresh_(nms_thresh),
      class_names_(std::move(class_names)),
      model_path_(model_path) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(2);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session_ = std::make_unique<Ort::Session>(env_, model_path_.c_str(), session_options);

    size_t in_count = session_->GetInputCount();
    if (in_count == 0) {
        throw std::runtime_error("Model has no inputs.");
    }
    input_name_ = session_->GetInputNameAllocated(0, *allocator_).get();

    size_t out_count = session_->GetOutputCount();
    if (out_count == 0) {
        throw std::runtime_error("Model has no outputs.");
    }
    output_name_ = session_->GetOutputNameAllocated(0, *allocator_).get();

    Ort::TypeInfo out_type_info = session_->GetOutputTypeInfo(0);
    auto tensor_info = out_type_info.GetTensorTypeAndShapeInfo();
    output_shape_ = tensor_info.GetShape();
}

std::vector<YoloResult> YoloInfer::infer(const std::string& image_path) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[WARN] YoloInfer::infer() failed to load image: " << image_path << "\n";
        return {};
    }
    return infer(img);
}

std::vector<YoloResult> YoloInfer::infer(const cv::Mat& image) {
    if (image.empty()) return {};

    int orig_w = image.cols;
    int orig_h = image.rows;

    std::vector<float> input_tensor_values;
    preprocess(image, input_tensor_values, input_w_, input_h_);
    std::array<int64_t, 4> input_shape = {1, 3, input_h_, input_w_};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = { input_name_.c_str() };
    const char* output_names[] = { output_name_.c_str() };

    auto output_tensors = session_->Run(Ort::RunOptions{ nullptr },
                                        input_names, &input_tensor, 1,
                                        output_names, 1);
    if (output_tensors.empty()) return {};

    auto& out_tensor = output_tensors[0];
    auto out_info = out_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = out_info.GetShape();

    if (shape.size() != 3) {
        if (shape.size() == 2) {
            shape.insert(shape.begin(), 1);
        } else {
            throw std::runtime_error("Unexpected output tensor shape rank (expected 3).");
        }
    }

    int64_t num_det = shape[1];
    int64_t elem_len = shape[2];

    bool transposed_output = false;
    if (elem_len > num_det) {
        // Ultralytics YOLOv8/11 exported ONNX uses [1, attrs, points].
        transposed_output = true;
        std::swap(num_det, elem_len);
    }

    int64_t expected_plain = 4 + static_cast<int64_t>(class_names_.size());
    int64_t expected_with_obj = 5 + static_cast<int64_t>(class_names_.size());
    bool has_objectness = (elem_len == expected_with_obj);
    bool plain_layout = (elem_len == expected_plain);
    if (!has_objectness && !plain_layout) {
        // Fall back to whichever layout elem_len resembles more.
        has_objectness = (elem_len > expected_plain);
        plain_layout = !has_objectness;
    }
    int cls_offset = has_objectness ? 5 : 4;

    float* out_data = out_tensor.GetTensorMutableData<float>();

    auto read_attr = [&](int attr_idx, int det_idx) -> float {
        if (transposed_output) {
            return out_data[attr_idx * num_det + det_idx];
        }
        return out_data[det_idx * elem_len + attr_idx];
    };

    enum class BoxEncoding { XYWH, XYXY };
    auto infer_box_encoding = [&](int64_t max_check) -> BoxEncoding {
        int64_t sample_count = std::min<int64_t>(num_det, max_check);
        if (sample_count <= 0) {
            return BoxEncoding::XYWH;
        }
        int xyxy_votes = 0;
        for (int64_t i = 0; i < sample_count; ++i) {
            float x0 = read_attr(0, static_cast<int>(i));
            float y0 = read_attr(1, static_cast<int>(i));
            float x1 = read_attr(2, static_cast<int>(i));
            float y1 = read_attr(3, static_cast<int>(i));
            if (x1 >= x0 && y1 >= y0) {
                xyxy_votes++;
            }
        }
        if (xyxy_votes * 2 >= sample_count) {
            return BoxEncoding::XYXY;
        }
        return BoxEncoding::XYWH;
    };
    BoxEncoding box_encoding = infer_box_encoding(64);

    std::vector<YoloResult> candidates;
    candidates.reserve(static_cast<size_t>(num_det));

    for (int64_t i = 0; i < num_det; ++i) {
        float raw0 = read_attr(0, static_cast<int>(i));
        float raw1 = read_attr(1, static_cast<int>(i));
        float raw2 = read_attr(2, static_cast<int>(i));
        float raw3 = read_attr(3, static_cast<int>(i));
        float obj_conf = has_objectness ? read_attr(4, static_cast<int>(i)) : 1.0f;

        float best_class_conf = 0.0f;
        int best_class_id = -1;
        for (int64_t c = cls_offset; c < elem_len; ++c) {
            float cls_conf = read_attr(static_cast<int>(c), static_cast<int>(i));
            if (cls_conf > best_class_conf) {
                best_class_conf = cls_conf;
                best_class_id = static_cast<int>(c - cls_offset);
            }
        }

        float final_conf = obj_conf * best_class_conf;
        if (final_conf < conf_thresh_) continue;

        cv::Rect box;
        if (box_encoding == BoxEncoding::XYXY) {
            float x0 = raw0;
            float y0 = raw1;
            float x1 = raw2;
            float y1 = raw3;
            if (x1 <= x0 || y1 <= y0) {
                continue;
            }
            box = cv::Rect(
                static_cast<int>(std::round(x0)),
                static_cast<int>(std::round(y0)),
                static_cast<int>(std::round(x1 - x0)),
                static_cast<int>(std::round(y1 - y0))
            );
        } else {
            float cx = raw0;
            float cy = raw1;
            float w  = raw2;
            float h  = raw3;
            float x = cx - w * 0.5f;
            float y = cy - h * 0.5f;
            box = cv::Rect(
                static_cast<int>(std::round(x)),
                static_cast<int>(std::round(y)),
                static_cast<int>(std::round(w)),
                static_cast<int>(std::round(h))
            );
        }

        cv::Rect box_orig = scale_coords_back(box, orig_w, orig_h, input_w_, input_h_);

        YoloResult r;
        r.class_id = best_class_id;
        r.score = final_conf;
        r.box = box_orig;
        candidates.push_back(r);
    }

    std::vector<int> keep = nms(candidates, nms_thresh_);
    std::vector<YoloResult> results;
    results.reserve(keep.size());
    for (int idx : keep) {
        results.push_back(candidates[idx]);
    }
    return results;
}

std::string YoloInfer::classNameOrDefault(int class_id) const {
    if (class_id >= 0 && class_id < static_cast<int>(class_names_.size())) {
        return class_names_[class_id];
    }
    return "class_" + std::to_string(class_id);
}

#ifdef YOLOINFER_DEMO_MAIN
int main(int argc, char** argv) {
    std::wstring model_path = kDefaultModelPath;
    if (argc > 1) {
        std::string arg1 = argv[1];
        std::wstring warg(arg1.begin(), arg1.end());
        model_path = warg;
    }

    YoloInfer infer(model_path);

    std::string img_path = "test.jpg";
    if (argc > 2) img_path = argv[2];

    auto results = infer.infer(img_path);
    if (results.empty()) {
        std::cout << "No detections found for " << img_path << "\n";
        return 0;
    }

    cv::Mat img = cv::imread(img_path);
    for (const auto& r : results) {
        cv::rectangle(img, r.box, cv::Scalar(0, 255, 0), 2);
        std::string label = infer.classNameOrDefault(r.class_id) + ":" + std::to_string(r.score);
        cv::putText(img, label, cv::Point(r.box.x, r.box.y - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    }
    cv::imwrite("out.jpg", img);
    std::cout << "Saved out.jpg\n";
    return 0;
}
#endif
