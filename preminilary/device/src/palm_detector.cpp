#include "../include/common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

namespace {

using PredictClock = std::chrono::steady_clock;

double PredictDurationMs(const PredictClock::time_point& begin,
                         const PredictClock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - begin).count();
}

void UpdateAccountedTiming(PalmPredictTiming* timing) {
    if (timing == nullptr) {
        return;
    }

    timing->accounted_ms = timing->preprocess_ms +
                           timing->input_load_ms +
                           timing->inference_ms +
                           timing->getoutput_ms +
                           timing->output_meta_ms +
                           timing->decode_ms +
                           timing->verbose_log_ms;
}

}  // namespace

void PalmResult::Clear() {
    detections.clear();
    valid = false;
}

PALMDETECTOR::PALMDETECTOR() {
    manual_input.data = nullptr;
    inputs[0].data = nullptr;
    for (int i = 0; i < kPalmOutputCount; i++) {
        outputs[i].data = nullptr;
    }
}

void PALMDETECTOR::Initialize(const std::string& model_path,
                              const std::array<int, 2>& in_image_shape,
                              const std::array<int, 2>& in_input_shape,
                              bool in_use_ai_preprocess,
                              bool in_rotate_clockwise,
                              PalmOutputLayout in_output_layout) {
    image_shape = in_image_shape;
    rotated_shape = {image_shape[1], image_shape[0]};
    input_shape = in_input_shape;
    use_ai_preprocess = in_use_ai_preprocess;
    rotate_clockwise = in_rotate_clockwise;
    output_layout = in_output_layout;
    pipe_offline = nullptr;

    char* model_path_char = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_char, SSNE_STATIC_ALLOC);
    std::cout << "[PALM] Loaded model: " << model_path << ", model_id=" << model_id << std::endl;

    const int input_num = ssne_get_model_input_num(model_id);
    int input_dtype = -1;
    ssne_get_model_input_dtype(model_id, &input_dtype);

    int mean[3] = {0, 0, 0};
    int std_scale[3] = {0, 0, 0};
    int is_uint8 = 0;
    const int norm_ret = ssne_get_model_normalize_params(model_id, mean, std_scale, &is_uint8);
    std::cout << "[PALM] Model input_num=" << input_num
              << ", input_dtype=" << input_dtype
              << ", normalize_ret=" << norm_ret
              << ", mean=(" << mean[0] << "," << mean[1] << "," << mean[2] << ")"
              << ", std=(" << std_scale[0] << "," << std_scale[1] << "," << std_scale[2] << ")"
              << ", is_uint8=" << is_uint8 << std::endl;

    manual_input = create_tensor(static_cast<uint32_t>(input_shape[0]),
                                 static_cast<uint32_t>(input_shape[1]),
                                 SSNE_Y_8,
                                 SSNE_BUF_AI);
    inputs[0] = create_tensor(static_cast<uint32_t>(input_shape[0]),
                              static_cast<uint32_t>(input_shape[1]),
                              SSNE_Y_8,
                              SSNE_BUF_AI);
    manual_input_buffer.resize(static_cast<size_t>(input_shape[0]) *
                               static_cast<size_t>(input_shape[1]));

    if (use_ai_preprocess) {
        pipe_offline = GetAIPreprocessPipe();
        const int set_norm_ret = SetNormalize(pipe_offline, model_id);
        if (set_norm_ret != 0) {
            std::cerr << "[PALM] Warning: SetNormalize failed, ret=" << set_norm_ret
                      << ". Manual 224x224 gray input will still be generated." << std::endl;
        } else {
            std::cout << "[PALM] AI preprocess normalization was configured from the model." << std::endl;
        }
    } else {
        std::cout << "[PALM] AI preprocess is bypassed. The 224x224 gray buffer will be copied"
                  << " directly into the model input tensor." << std::endl;
    }

    const TensorDebugInfo manual_info = GetTensorDebugInfo(manual_input);
    const TensorDebugInfo input_info = GetTensorDebugInfo(inputs[0]);
    std::cout << "[PALM] Manual preprocessing: camera=" << image_shape[0] << "x" << image_shape[1];
    if (rotate_clockwise) {
        std::cout << " -> clockwise rotated=" << rotated_shape[0] << "x" << rotated_shape[1];
    }
    std::cout << " -> bilinear resized=" << input_shape[0] << "x" << input_shape[1] << std::endl;
    std::cout << "[PALM] Detector anchors: head14=" << (kPalmFeature14 * kPalmFeature14 * kPalmNumAnchorsPerCell)
              << ", head7=" << (kPalmFeature7 * kPalmFeature7 * kPalmNumAnchorsPerCell)
              << ", score_threshold=" << kPalmScoreThreshold
              << ", nms_iou=" << kPalmNmsIouThreshold
              << ", max_detections=" << kPalmMaxDetections << std::endl;
    std::cout << "[PALM] Active output layout=" << OutputLayoutName(output_layout) << std::endl;
    std::cout << "[PALM] Manual input tensor: width=" << manual_info.width
              << ", height=" << manual_info.height
              << ", dtype=" << static_cast<int>(manual_info.dtype)
              << ", format=" << static_cast<int>(manual_info.format)
              << ", mem_size=" << manual_info.mem_size
              << ", total_size=" << manual_info.total_size << std::endl;
    std::cout << "[PALM] Model input tensor: width=" << input_info.width
              << ", height=" << input_info.height
              << ", dtype=" << static_cast<int>(input_info.dtype)
              << ", format=" << static_cast<int>(input_info.format)
              << ", mem_size=" << input_info.mem_size
              << ", total_size=" << input_info.total_size << std::endl;

    initialized = true;
}

void PALMDETECTOR::Predict(ssne_tensor_t* img,
                           PalmResult* result,
                           uint32_t frame_index,
                           bool verbose_log,
                           PalmPredictTiming* timing) {
    if (timing != nullptr) {
        *timing = PalmPredictTiming();
    }

    result->Clear();
    if (!initialized) {
        std::cerr << "[PALM] Predict called before Initialize." << std::endl;
        return;
    }

    PredictClock::time_point stage_begin = PredictClock::now();
    if (!PreprocessRotateResize(*img, timing)) {
        if (timing != nullptr) {
            timing->preprocess_ms = PredictDurationMs(stage_begin, PredictClock::now());
            UpdateAccountedTiming(timing);
        }
        return;
    }
    if (timing != nullptr) {
        timing->preprocess_ms = PredictDurationMs(stage_begin, PredictClock::now());
    }

    stage_begin = PredictClock::now();
    if (use_ai_preprocess) {
        if (pipe_offline == nullptr) {
            std::cerr << "[PALM] AI preprocess is enabled but pipe_offline is null." << std::endl;
            if (timing != nullptr) {
                timing->input_load_ms = PredictDurationMs(stage_begin, PredictClock::now());
                UpdateAccountedTiming(timing);
            }
            return;
        }

        const int preprocess_ret = RunAiPreprocessPipe(pipe_offline, manual_input, inputs[0]);
        if (preprocess_ret != 0) {
            std::cerr << "[PALM] RunAiPreprocessPipe failed, ret=" << preprocess_ret << std::endl;
            if (timing != nullptr) {
                timing->input_load_ms = PredictDurationMs(stage_begin, PredictClock::now());
                UpdateAccountedTiming(timing);
            }
            return;
        }
    } else {
        const int direct_ret = load_tensor_buffer_ptr(inputs[0],
                                                     manual_input_buffer.data(),
                                                     static_cast<int>(manual_input_buffer.size()));
        if (direct_ret != 0) {
            std::cerr << "[PALM] Direct load_tensor_buffer_ptr for model input failed, ret="
                      << direct_ret << std::endl;
            if (timing != nullptr) {
                timing->input_load_ms = PredictDurationMs(stage_begin, PredictClock::now());
                UpdateAccountedTiming(timing);
            }
            return;
        }
    }
    if (timing != nullptr) {
        timing->input_load_ms = PredictDurationMs(stage_begin, PredictClock::now());
    }

    stage_begin = PredictClock::now();
    const int infer_ret = ssne_inference(model_id, 1, inputs);
    if (timing != nullptr) {
        timing->inference_ms = PredictDurationMs(stage_begin, PredictClock::now());
    }
    if (infer_ret != 0) {
        std::cerr << "[PALM] ssne_inference failed, ret=" << infer_ret << std::endl;
        UpdateAccountedTiming(timing);
        return;
    }

    stage_begin = PredictClock::now();
    const int output_ret = ssne_getoutput(model_id, kPalmOutputCount, outputs);
    if (timing != nullptr) {
        timing->getoutput_ms = PredictDurationMs(stage_begin, PredictClock::now());
    }
    if (output_ret != 0) {
        std::cerr << "[PALM] ssne_getoutput failed, ret=" << output_ret << std::endl;
        UpdateAccountedTiming(timing);
        return;
    }

    stage_begin = PredictClock::now();
    TensorDebugInfo output_info[kPalmOutputCount];
    for (int i = 0; i < kPalmOutputCount; i++) {
        output_info[i] = GetTensorDebugInfo(outputs[i]);
    }

    const OutputMapping mapping = MapOutputs(output_info);
    if (timing != nullptr) {
        timing->output_meta_ms = PredictDurationMs(stage_begin, PredictClock::now());
    }

    stage_begin = PredictClock::now();
    DecodeOutputs(mapping, output_info, output_layout, result);
    if (timing != nullptr) {
        timing->decode_ms = PredictDurationMs(stage_begin, PredictClock::now());
    }

    if (verbose_log) {
        stage_begin = PredictClock::now();
        const TensorDebugInfo manual_input_info = GetTensorDebugInfo(manual_input);
        const TensorDebugInfo input_info = GetTensorDebugInfo(inputs[0]);
        PrintFrameLog(frame_index,
                      *img,
                      manual_input_info,
                      input_info,
                      output_info,
                      mapping,
                      *result);
        if (timing != nullptr) {
            timing->verbose_log_ms = PredictDurationMs(stage_begin, PredictClock::now());
        }
    }

    if (timing != nullptr) {
        timing->success = true;
        UpdateAccountedTiming(timing);
    }
}

void PALMDETECTOR::Release() {
    if (manual_input.data != nullptr) {
        release_tensor(manual_input);
        manual_input.data = nullptr;
    }

    if (inputs[0].data != nullptr) {
        release_tensor(inputs[0]);
        inputs[0].data = nullptr;
    }

    for (int i = 0; i < kPalmOutputCount; i++) {
        if (outputs[i].data != nullptr) {
            release_tensor(outputs[i]);
            outputs[i].data = nullptr;
        }
    }

    if (pipe_offline != nullptr) {
        ReleaseAIPreprocessPipe(pipe_offline);
        pipe_offline = nullptr;
    }

    initialized = false;
}

float PALMDETECTOR::Clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

PALMDETECTOR::TensorDebugInfo PALMDETECTOR::GetTensorDebugInfo(ssne_tensor_t tensor) {
    TensorDebugInfo info;
    info.width = get_width(tensor);
    info.height = get_height(tensor);
    info.dtype = get_data_type(tensor);
    info.format = get_data_format(tensor);
    info.mem_size = get_mem_size(tensor);
    info.total_size = get_total_size(tensor);
    info.inferred_elements = InferElementCount(info);
    return info;
}

PALMDETECTOR::TensorValueStats PALMDETECTOR::GetTensorValueStats(ssne_tensor_t tensor,
                                                                 const TensorDebugInfo& info) {
    TensorValueStats stats;
    const void* data = get_data(tensor);
    if (data == nullptr || info.inferred_elements == 0) {
        return stats;
    }

    stats.element_count = info.inferred_elements;
    double sum_value = 0.0;
    double min_value = std::numeric_limits<double>::max();
    double max_value = std::numeric_limits<double>::lowest();
    stats.sample_first = ReadTensorValue(tensor, info, 0);
    stats.sample_center = ReadTensorValue(tensor, info, stats.element_count / 2);
    stats.sample_last = ReadTensorValue(tensor, info, stats.element_count - 1);

    for (size_t i = 0; i < stats.element_count; i++) {
        const double value = static_cast<double>(ReadTensorValue(tensor, info, i));
        if (!std::isfinite(value)) {
            continue;
        }
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        sum_value += value;
        stats.finite_count += 1;
        stats.nonzero_count += value != 0.0 ? 1 : 0;
        stats.near_half_count += (value >= 0.49 && value <= 0.51) ? 1 : 0;
    }

    if (stats.finite_count == 0) {
        return stats;
    }

    stats.valid = true;
    stats.min_value = min_value;
    stats.max_value = max_value;
    stats.mean_value = sum_value / static_cast<double>(stats.finite_count);
    return stats;
}

size_t PALMDETECTOR::InferElementCount(const TensorDebugInfo& info) {
    if (info.dtype == SSNE_FLOAT32) {
        return info.mem_size / sizeof(float);
    }
    if (info.dtype == SSNE_UINT8 || info.dtype == SSNE_INT8) {
        return info.mem_size;
    }
    return info.total_size;
}

float PALMDETECTOR::ReadTensorValue(ssne_tensor_t tensor,
                                    const TensorDebugInfo& info,
                                    size_t index) {
    const void* data = get_data(tensor);
    if (data == nullptr || index >= info.inferred_elements) {
        return 0.0f;
    }

    if (info.dtype == SSNE_FLOAT32) {
        const float* values = reinterpret_cast<const float*>(data);
        return values[index];
    }
    if (info.dtype == SSNE_UINT8) {
        const uint8_t* values = reinterpret_cast<const uint8_t*>(data);
        return static_cast<float>(values[index]);
    }
    if (info.dtype == SSNE_INT8) {
        const int8_t* values = reinterpret_cast<const int8_t*>(data);
        return static_cast<float>(values[index]);
    }
    return 0.0f;
}

float PALMDETECTOR::IoU(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    const float xx1 = std::max(a[0], b[0]);
    const float yy1 = std::max(a[1], b[1]);
    const float xx2 = std::min(a[2], b[2]);
    const float yy2 = std::min(a[3], b[3]);
    const float inter_w = std::max(0.0f, xx2 - xx1);
    const float inter_h = std::max(0.0f, yy2 - yy1);
    const float inter = inter_w * inter_h;
    const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    return inter / (area_a + area_b - inter + 1e-6f);
}

std::vector<int> PALMDETECTOR::NmsIndices(const std::vector<Candidate>& candidates, float iou_threshold) {
    std::vector<int> order;
    for (size_t i = 0; i < candidates.size(); i++) {
        order.push_back(static_cast<int>(i));
    }
    std::sort(order.begin(), order.end(), [&candidates](int a, int b) {
        return candidates[a].detection.score > candidates[b].detection.score;
    });

    std::vector<int> keep;
    while (!order.empty()) {
        const int current = order.front();
        keep.push_back(current);

        std::vector<int> remaining;
        for (size_t i = 1; i < order.size(); i++) {
            const int idx = order[i];
            const float iou = IoU(candidates[current].detection.model_box,
                                  candidates[idx].detection.model_box);
            if (iou <= iou_threshold) {
                remaining.push_back(idx);
            }
        }
        order.swap(remaining);
    }
    return keep;
}

bool PALMDETECTOR::IsSameIndexUsed(const std::vector<int>& values, int value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

size_t PALMDETECTOR::ExpectedRegElements(int feature_size) {
    return static_cast<size_t>(feature_size) *
           static_cast<size_t>(feature_size) *
           static_cast<size_t>(kPalmRegChannels);
}

size_t PALMDETECTOR::ExpectedClsElements(int feature_size) {
    return static_cast<size_t>(feature_size) *
           static_cast<size_t>(feature_size) *
           static_cast<size_t>(kPalmClsChannels);
}

const char* PALMDETECTOR::OutputLayoutName(PalmOutputLayout layout) {
    return layout == kPalmOutputLayoutHwc ? "HWC" : "NCHW";
}

PALMDETECTOR::OutputMapping PALMDETECTOR::MapOutputs(const TensorDebugInfo output_info[kPalmOutputCount]) const {
    OutputMapping mapping;
    std::vector<int> used;

    const size_t expected_values[kPalmOutputCount] = {
        ExpectedRegElements(kPalmFeature14),
        ExpectedClsElements(kPalmFeature14),
        ExpectedRegElements(kPalmFeature7),
        ExpectedClsElements(kPalmFeature7),
    };
    int* slots[kPalmOutputCount] = {
        &mapping.reg14,
        &mapping.cls14,
        &mapping.reg7,
        &mapping.cls7,
    };

    for (int expected_idx = 0; expected_idx < kPalmOutputCount; expected_idx++) {
        for (int out_idx = 0; out_idx < kPalmOutputCount; out_idx++) {
            if (IsSameIndexUsed(used, out_idx)) {
                continue;
            }
            if (output_info[out_idx].inferred_elements == expected_values[expected_idx]) {
                *slots[expected_idx] = out_idx;
                used.push_back(out_idx);
                break;
            }
        }
    }

    mapping.valid = mapping.reg14 >= 0 && mapping.cls14 >= 0 &&
                    mapping.reg7 >= 0 && mapping.cls7 >= 0;
    if (mapping.valid) {
        mapping.reason = "matched outputs by expected element counts";
        return mapping;
    }

    mapping.reg14 = 0;
    mapping.cls14 = 1;
    mapping.reg7 = 2;
    mapping.cls7 = 3;
    mapping.valid = false;
    mapping.reason = "could not fully match by element counts; fallback to Keras output order";
    return mapping;
}

bool PALMDETECTOR::PreprocessRotateResize(ssne_tensor_t camera_tensor, PalmPredictTiming* timing) {
    const uint8_t* camera_data = reinterpret_cast<const uint8_t*>(get_data(camera_tensor));
    if (camera_data == nullptr) {
        std::cerr << "[PALM] Camera tensor has null data." << std::endl;
        return false;
    }

    const TensorDebugInfo camera_info = GetTensorDebugInfo(camera_tensor);
    const int src_width = camera_info.width > 0 ? static_cast<int>(camera_info.width) : image_shape[0];
    const int src_height = camera_info.height > 0 ? static_cast<int>(camera_info.height) : image_shape[1];

    if (camera_info.format != SSNE_Y_8) {
        std::cerr << "[PALM] Warning: expected SSNE_Y_8 camera tensor, got format="
                  << static_cast<int>(camera_info.format) << std::endl;
    }

    PredictClock::time_point stage_begin = PredictClock::now();
    if (rotate_clockwise) {
        ResizeClockwiseRotatedBilinear(camera_data,
                                       src_width,
                                       src_height,
                                       manual_input_buffer.data(),
                                       input_shape[0],
                                       input_shape[1]);
    } else {
        ResizeBilinear(camera_data,
                       src_width,
                       src_height,
                       manual_input_buffer.data(),
                       input_shape[0],
                       input_shape[1]);
    }
    if (timing != nullptr) {
        timing->preprocess_transform_ms = PredictDurationMs(stage_begin, PredictClock::now());
    }

    stage_begin = PredictClock::now();
    const int ret = load_tensor_buffer_ptr(manual_input,
                                           manual_input_buffer.data(),
                                           static_cast<int>(manual_input_buffer.size()));
    if (timing != nullptr) {
        timing->preprocess_manual_load_ms = PredictDurationMs(stage_begin, PredictClock::now());
    }
    if (ret != 0) {
        std::cerr << "[PALM] load_tensor_buffer_ptr for manual input failed, ret=" << ret << std::endl;
        return false;
    }

    return true;
}

void PALMDETECTOR::ResizeBilinear(const uint8_t* src,
                                  int src_width,
                                  int src_height,
                                  uint8_t* dst,
                                  int dst_width,
                                  int dst_height) const {
    const double scale_x = static_cast<double>(src_width) / static_cast<double>(dst_width);
    const double scale_y = static_cast<double>(src_height) / static_cast<double>(dst_height);

    for (int dst_y = 0; dst_y < dst_height; dst_y++) {
        const double src_y = (static_cast<double>(dst_y) + 0.5) * scale_y - 0.5;
        int y0 = static_cast<int>(std::floor(src_y));
        double wy = src_y - static_cast<double>(y0);
        if (y0 < 0) {
            y0 = 0;
            wy = 0.0;
        }
        int y1 = y0 + 1;
        if (y1 >= src_height) {
            y1 = src_height - 1;
            y0 = y1;
            wy = 0.0;
        }

        for (int dst_x = 0; dst_x < dst_width; dst_x++) {
            const double src_x = (static_cast<double>(dst_x) + 0.5) * scale_x - 0.5;
            int x0 = static_cast<int>(std::floor(src_x));
            double wx = src_x - static_cast<double>(x0);
            if (x0 < 0) {
                x0 = 0;
                wx = 0.0;
            }
            int x1 = x0 + 1;
            if (x1 >= src_width) {
                x1 = src_width - 1;
                x0 = x1;
                wx = 0.0;
            }

            const double v00 = static_cast<double>(src[y0 * src_width + x0]);
            const double v01 = static_cast<double>(src[y0 * src_width + x1]);
            const double v10 = static_cast<double>(src[y1 * src_width + x0]);
            const double v11 = static_cast<double>(src[y1 * src_width + x1]);
            const double top = v00 * (1.0 - wx) + v01 * wx;
            const double bottom = v10 * (1.0 - wx) + v11 * wx;
            const double value = top * (1.0 - wy) + bottom * wy;
            dst[dst_y * dst_width + dst_x] =
                static_cast<uint8_t>(std::max(0.0, std::min(255.0, std::round(value))));
        }
    }
}

void PALMDETECTOR::ResizeClockwiseRotatedBilinear(const uint8_t* src,
                                                  int src_width,
                                                  int src_height,
                                                  uint8_t* dst,
                                                  int dst_width,
                                                  int dst_height) const {
    const int rotated_width = src_height;
    const int rotated_height = src_width;
    const double scale_x = static_cast<double>(rotated_width) / static_cast<double>(dst_width);
    const double scale_y = static_cast<double>(rotated_height) / static_cast<double>(dst_height);

    for (int dst_y = 0; dst_y < dst_height; dst_y++) {
        const double rot_yf = (static_cast<double>(dst_y) + 0.5) * scale_y - 0.5;
        int rot_y0 = static_cast<int>(std::floor(rot_yf));
        double wy = rot_yf - static_cast<double>(rot_y0);
        if (rot_y0 < 0) {
            rot_y0 = 0;
            wy = 0.0;
        }
        int rot_y1 = rot_y0 + 1;
        if (rot_y1 >= rotated_height) {
            rot_y1 = rotated_height - 1;
            rot_y0 = rot_y1;
            wy = 0.0;
        }

        for (int dst_x = 0; dst_x < dst_width; dst_x++) {
            const double rot_xf = (static_cast<double>(dst_x) + 0.5) * scale_x - 0.5;
            int rot_x0 = static_cast<int>(std::floor(rot_xf));
            double wx = rot_xf - static_cast<double>(rot_x0);
            if (rot_x0 < 0) {
                rot_x0 = 0;
                wx = 0.0;
            }
            int rot_x1 = rot_x0 + 1;
            if (rot_x1 >= rotated_width) {
                rot_x1 = rotated_width - 1;
                rot_x0 = rot_x1;
                wx = 0.0;
            }

            const int src_x00 = rot_y0;
            const int src_y00 = src_height - 1 - rot_x0;
            const int src_x01 = rot_y0;
            const int src_y01 = src_height - 1 - rot_x1;
            const int src_x10 = rot_y1;
            const int src_y10 = src_height - 1 - rot_x0;
            const int src_x11 = rot_y1;
            const int src_y11 = src_height - 1 - rot_x1;

            const double v00 = static_cast<double>(src[src_y00 * src_width + src_x00]);
            const double v01 = static_cast<double>(src[src_y01 * src_width + src_x01]);
            const double v10 = static_cast<double>(src[src_y10 * src_width + src_x10]);
            const double v11 = static_cast<double>(src[src_y11 * src_width + src_x11]);

            const double top = v00 * (1.0 - wx) + v01 * wx;
            const double bottom = v10 * (1.0 - wx) + v11 * wx;
            const double value = top * (1.0 - wy) + bottom * wy;
            dst[dst_y * dst_width + dst_x] =
                static_cast<uint8_t>(std::max(0.0, std::min(255.0, std::round(value))));
        }
    }
}

PALMDETECTOR::Anchor PALMDETECTOR::GetAnchor(int feature_size,
                                             int cell_x,
                                             int cell_y,
                                             int anchor_index) const {
    Anchor anchor;
    const float step = 1.0f / static_cast<float>(feature_size);
    anchor.cx = static_cast<float>(cell_x) * step + step * 0.5f;
    anchor.cy = static_cast<float>(cell_y) * step + step * 0.5f;

    if (feature_size == kPalmFeature14) {
        anchor.w = anchor_index == 0 ? 0.10f : 0.18f;
        anchor.h = anchor_index == 0 ? 0.10f : 0.18f;
    } else {
        anchor.w = anchor_index == 0 ? 0.25f : 0.40f;
        anchor.h = anchor_index == 0 ? 0.25f : 0.40f;
    }
    return anchor;
}

PalmKeypoint PALMDETECTOR::MapPoint(float model_x, float model_y) const {
    PalmKeypoint point;
    point.model_x = Clamp01(model_x);
    point.model_y = Clamp01(model_y);

    float original_x = 0.0f;
    float original_y = 0.0f;
    if (rotate_clockwise) {
        const float rotated_x = point.model_x * static_cast<float>(rotated_shape[0] - 1);
        const float rotated_y = point.model_y * static_cast<float>(rotated_shape[1] - 1);
        original_x = rotated_y;
        original_y = static_cast<float>(image_shape[1] - 1) - rotated_x;
    } else {
        original_x = point.model_x * static_cast<float>(image_shape[0] - 1);
        original_y = point.model_y * static_cast<float>(image_shape[1] - 1);
    }

    original_x = std::max(0.0f, std::min(original_x, static_cast<float>(image_shape[0] - 1)));
    original_y = std::max(0.0f, std::min(original_y, static_cast<float>(image_shape[1] - 1)));
    point.pixel_x = static_cast<int>(std::round(original_x));
    point.pixel_y = static_cast<int>(std::round(original_y));
    point.x = static_cast<float>(point.pixel_x) / static_cast<float>(image_shape[0] - 1);
    point.y = static_cast<float>(point.pixel_y) / static_cast<float>(image_shape[1] - 1);
    return point;
}

std::array<float, 4> PALMDETECTOR::MapBox(const std::array<float, 4>& model_box) const {
    const PalmKeypoint p0 = MapPoint(model_box[0], model_box[1]);
    const PalmKeypoint p1 = MapPoint(model_box[2], model_box[1]);
    const PalmKeypoint p2 = MapPoint(model_box[2], model_box[3]);
    const PalmKeypoint p3 = MapPoint(model_box[0], model_box[3]);

    const float min_x = static_cast<float>(std::min(std::min(p0.pixel_x, p1.pixel_x),
                                                    std::min(p2.pixel_x, p3.pixel_x)));
    const float max_x = static_cast<float>(std::max(std::max(p0.pixel_x, p1.pixel_x),
                                                    std::max(p2.pixel_x, p3.pixel_x)));
    const float min_y = static_cast<float>(std::min(std::min(p0.pixel_y, p1.pixel_y),
                                                    std::min(p2.pixel_y, p3.pixel_y)));
    const float max_y = static_cast<float>(std::max(std::max(p0.pixel_y, p1.pixel_y),
                                                    std::max(p2.pixel_y, p3.pixel_y)));
    return std::array<float, 4>{{min_x, min_y, max_x, max_y}};
}

size_t PALMDETECTOR::GetOutputIndex(int feature_size,
                                    int channel_count,
                                    int channel,
                                    int cell_x,
                                    int cell_y,
                                    PalmOutputLayout layout) const {
    const size_t spatial = static_cast<size_t>(feature_size) * static_cast<size_t>(feature_size);
    const size_t cell_index = static_cast<size_t>(cell_y) * static_cast<size_t>(feature_size) +
                              static_cast<size_t>(cell_x);
    if (layout == kPalmOutputLayoutHwc) {
        return cell_index * static_cast<size_t>(channel_count) + static_cast<size_t>(channel);
    }
    return static_cast<size_t>(channel) * spatial + cell_index;
}

void PALMDETECTOR::DecodeHead(ssne_tensor_t reg_tensor,
                              const TensorDebugInfo& reg_info,
                              ssne_tensor_t cls_tensor,
                              const TensorDebugInfo& cls_info,
                              int feature_size,
                              PalmOutputLayout layout,
                              std::vector<Candidate>* candidates) const {
    const size_t expected_reg = ExpectedRegElements(feature_size);
    const size_t expected_cls = ExpectedClsElements(feature_size);
    if (reg_info.inferred_elements < expected_reg || cls_info.inferred_elements < expected_cls) {
        std::cerr << "[PALM] Output element count too small for head" << feature_size
                  << ": reg=" << reg_info.inferred_elements << "/" << expected_reg
                  << ", cls=" << cls_info.inferred_elements << "/" << expected_cls << std::endl;
        return;
    }

    for (int y = 0; y < feature_size; y++) {
        for (int x = 0; x < feature_size; x++) {
            for (int anchor_idx = 0; anchor_idx < kPalmNumAnchorsPerCell; anchor_idx++) {
                const size_t cls_index =
                    GetOutputIndex(feature_size, kPalmClsChannels, anchor_idx, x, y, layout);
                const float score = ReadTensorValue(cls_tensor, cls_info, cls_index);
                if (score < kPalmScoreThreshold || !std::isfinite(score)) {
                    continue;
                }

                const Anchor anchor = GetAnchor(feature_size, x, y, anchor_idx);
                const int reg_channel_base = anchor_idx * kPalmValuesPerAnchor;
                const float dx = ReadTensorValue(reg_tensor,
                                                 reg_info,
                                                 GetOutputIndex(feature_size,
                                                                kPalmRegChannels,
                                                                reg_channel_base + 0,
                                                                x,
                                                                y,
                                                                layout));
                const float dy = ReadTensorValue(reg_tensor,
                                                 reg_info,
                                                 GetOutputIndex(feature_size,
                                                                kPalmRegChannels,
                                                                reg_channel_base + 1,
                                                                x,
                                                                y,
                                                                layout));
                const float dw = ReadTensorValue(reg_tensor,
                                                 reg_info,
                                                 GetOutputIndex(feature_size,
                                                                kPalmRegChannels,
                                                                reg_channel_base + 2,
                                                                x,
                                                                y,
                                                                layout));
                const float dh = ReadTensorValue(reg_tensor,
                                                 reg_info,
                                                 GetOutputIndex(feature_size,
                                                                kPalmRegChannels,
                                                                reg_channel_base + 3,
                                                                x,
                                                                y,
                                                                layout));

                if (!std::isfinite(dx) || !std::isfinite(dy) ||
                    !std::isfinite(dw) || !std::isfinite(dh)) {
                    continue;
                }

                const float cx = anchor.cx + dx * anchor.w;
                const float cy = anchor.cy + dy * anchor.h;
                const float box_w = anchor.w * std::exp(std::max(-10.0f, std::min(10.0f, dw)));
                const float box_h = anchor.h * std::exp(std::max(-10.0f, std::min(10.0f, dh)));

                PalmDetection detection;
                detection.model_box[0] = Clamp01(cx - box_w * 0.5f);
                detection.model_box[1] = Clamp01(cy - box_h * 0.5f);
                detection.model_box[2] = Clamp01(cx + box_w * 0.5f);
                detection.model_box[3] = Clamp01(cy + box_h * 0.5f);
                detection.pixel_box = MapBox(detection.model_box);
                detection.score = score;
                detection.head_feature_size = feature_size;

                for (int kp = 0; kp < kPalmNumKeypoints; kp++) {
                    const int kx_channel = anchor_idx * kPalmValuesPerAnchor + 4 + kp * 2;
                    const int ky_channel = anchor_idx * kPalmValuesPerAnchor + 5 + kp * 2;
                    const float kx =
                        anchor.cx +
                        ReadTensorValue(reg_tensor,
                                        reg_info,
                                        GetOutputIndex(feature_size,
                                                       kPalmRegChannels,
                                                       kx_channel,
                                                       x,
                                                       y,
                                                       layout)) *
                            anchor.w;
                    const float ky =
                        anchor.cy +
                        ReadTensorValue(reg_tensor,
                                        reg_info,
                                        GetOutputIndex(feature_size,
                                                       kPalmRegChannels,
                                                       ky_channel,
                                                       x,
                                                       y,
                                                       layout)) *
                            anchor.h;
                    detection.keypoints[kp] = MapPoint(kx, ky);
                }

                Candidate candidate;
                candidate.detection = detection;
                candidate.original_index = static_cast<int>(candidates->size());
                candidates->push_back(candidate);
            }
        }
    }
}

void PALMDETECTOR::SelectDetections(const std::vector<Candidate>& candidates, PalmResult* result) const {
    std::vector<Candidate> head14;
    std::vector<Candidate> head7;
    for (size_t i = 0; i < candidates.size(); i++) {
        if (candidates[i].detection.head_feature_size == kPalmFeature14) {
            head14.push_back(candidates[i]);
        } else {
            head7.push_back(candidates[i]);
        }
    }

    std::vector<PalmDetection> selected;
    const std::vector<int> keep14 = NmsIndices(head14, kPalmNmsIouThreshold);
    for (size_t i = 0; i < keep14.size(); i++) {
        selected.push_back(head14[keep14[i]].detection);
    }

    std::vector<int> order7;
    for (size_t i = 0; i < head7.size(); i++) {
        order7.push_back(static_cast<int>(i));
    }
    std::sort(order7.begin(), order7.end(), [&head7](int a, int b) {
        return head7[a].detection.score > head7[b].detection.score;
    });

    for (size_t i = 0; i < order7.size(); i++) {
        const PalmDetection& cand = head7[order7[i]].detection;
        bool suppress = false;
        for (size_t j = 0; j < selected.size(); j++) {
            if (IoU(cand.model_box, selected[j].model_box) > kPalmCrossHeadSuppressIou) {
                suppress = true;
                break;
            }
        }
        if (suppress) {
            continue;
        }
        selected.push_back(cand);
        if (static_cast<int>(selected.size()) >= kPalmMaxDetections) {
            break;
        }
    }

    if (selected.empty() && !candidates.empty()) {
        const std::vector<int> keep = NmsIndices(candidates, kPalmNmsIouThreshold);
        for (size_t i = 0; i < keep.size() && static_cast<int>(selected.size()) < kPalmMaxDetections; i++) {
            selected.push_back(candidates[keep[i]].detection);
        }
    }

    std::sort(selected.begin(), selected.end(), [](const PalmDetection& a, const PalmDetection& b) {
        return a.score > b.score;
    });
    if (static_cast<int>(selected.size()) > kPalmMaxDetections) {
        selected.resize(kPalmMaxDetections);
    }

    result->detections = selected;
    result->valid = !result->detections.empty();
}

void PALMDETECTOR::DecodeOutputs(const OutputMapping& mapping,
                                 const TensorDebugInfo output_info[kPalmOutputCount],
                                 PalmOutputLayout layout,
                                 PalmResult* result) const {
    result->Clear();
    if (!mapping.valid) {
        return;
    }

    std::vector<Candidate> candidates;
    DecodeHead(outputs[mapping.reg14],
               output_info[mapping.reg14],
               outputs[mapping.cls14],
               output_info[mapping.cls14],
               kPalmFeature14,
               layout,
               &candidates);
    DecodeHead(outputs[mapping.reg7],
               output_info[mapping.reg7],
               outputs[mapping.cls7],
               output_info[mapping.cls7],
               kPalmFeature7,
               layout,
               &candidates);
    SelectDetections(candidates, result);
}

void PALMDETECTOR::PrintTensorValueStats(uint32_t frame_index,
                                         const std::string& label,
                                         ssne_tensor_t tensor,
                                         const TensorDebugInfo& info) const {
    const TensorValueStats stats = GetTensorValueStats(tensor, info);
    if (!stats.valid) {
        std::cout << "[PALM][frame " << frame_index << "] " << label
                  << " value stats: unavailable, dtype=" << static_cast<int>(info.dtype)
                  << ", inferred_elements=" << info.inferred_elements
                  << ", data_ptr=" << get_data(tensor) << std::endl;
        return;
    }

    std::cout << "[PALM][frame " << frame_index << "] " << label
              << " value stats: dtype=" << static_cast<int>(info.dtype)
              << ", elements=" << stats.element_count
              << ", finite=" << stats.finite_count << "/" << stats.element_count
              << ", min=" << stats.min_value
              << ", max=" << stats.max_value
              << ", mean=" << stats.mean_value
              << ", nonzero=" << stats.nonzero_count << "/" << stats.element_count
              << ", near_0.5_count=" << stats.near_half_count << "/" << stats.element_count
              << ", sample_first=" << stats.sample_first
              << ", sample_center=" << stats.sample_center
              << ", sample_last=" << stats.sample_last
              << std::endl;
}

void PALMDETECTOR::PrintFrameLog(uint32_t frame_index,
                                 ssne_tensor_t camera_tensor,
                                 const TensorDebugInfo& manual_input_info,
                                 const TensorDebugInfo& model_input_info,
                                 const TensorDebugInfo output_info[kPalmOutputCount],
                                 const OutputMapping& mapping,
                                 const PalmResult& result) const {
    const TensorDebugInfo camera_info = GetTensorDebugInfo(camera_tensor);

    std::cout << "[PALM][frame " << frame_index << "] camera tensor: width=" << camera_info.width
              << ", height=" << camera_info.height
              << ", dtype=" << static_cast<int>(camera_info.dtype)
              << ", format=" << static_cast<int>(camera_info.format)
              << ", mem_size=" << camera_info.mem_size
              << ", total_size=" << camera_info.total_size
              << ", inferred_elements=" << camera_info.inferred_elements << std::endl;

    std::cout << "[PALM][frame " << frame_index << "] preprocessing: original="
              << image_shape[0] << "x" << image_shape[1]
              << ", rotate_clockwise=" << (rotate_clockwise ? 1 : 0)
              << ", resized=" << input_shape[0] << "x" << input_shape[1]
              << ", resize=bilinear" << std::endl;

    std::cout << "[PALM][frame " << frame_index << "] manual input tensor: width="
              << manual_input_info.width
              << ", height=" << manual_input_info.height
              << ", dtype=" << static_cast<int>(manual_input_info.dtype)
              << ", format=" << static_cast<int>(manual_input_info.format)
              << ", mem_size=" << manual_input_info.mem_size
              << ", total_size=" << manual_input_info.total_size
              << ", inferred_elements=" << manual_input_info.inferred_elements << std::endl;

    if (!manual_input_buffer.empty()) {
        uint8_t min_value = 255;
        uint8_t max_value = 0;
        uint64_t sum_value = 0;
        size_t nonzero_count = 0;
        for (size_t i = 0; i < manual_input_buffer.size(); i++) {
            const uint8_t value = manual_input_buffer[i];
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            sum_value += static_cast<uint64_t>(value);
            nonzero_count += value != 0 ? 1 : 0;
        }

        const double mean_value =
            static_cast<double>(sum_value) / static_cast<double>(manual_input_buffer.size());
        std::cout << "[PALM][frame " << frame_index << "] manual_input_buffer stats: size="
                  << manual_input_buffer.size()
                  << ", min=" << static_cast<int>(min_value)
                  << ", max=" << static_cast<int>(max_value)
                  << ", mean=" << mean_value
                  << ", nonzero=" << nonzero_count << "/" << manual_input_buffer.size()
                  << std::endl;
    }

    PrintTensorValueStats(frame_index, "manual_input_tensor_before_ai_preprocess", manual_input, manual_input_info);

    std::cout << "[PALM][frame " << frame_index << "] model input tensor: width=" << model_input_info.width
              << ", height=" << model_input_info.height
              << ", dtype=" << static_cast<int>(model_input_info.dtype)
              << ", format=" << static_cast<int>(model_input_info.format)
              << ", mem_size=" << model_input_info.mem_size
              << ", total_size=" << model_input_info.total_size
              << ", inferred_elements=" << model_input_info.inferred_elements << std::endl;
    PrintTensorValueStats(frame_index,
                          use_ai_preprocess ? "model_input_after_ai_preprocess"
                                            : "model_input_direct_from_manual_buffer",
                          inputs[0],
                          model_input_info);

    for (int i = 0; i < kPalmOutputCount; i++) {
        std::cout << "[PALM][frame " << frame_index << "] output[" << i << "]: width=" << output_info[i].width
                  << ", height=" << output_info[i].height
                  << ", dtype=" << static_cast<int>(output_info[i].dtype)
                  << ", format=" << static_cast<int>(output_info[i].format)
                  << ", mem_size=" << output_info[i].mem_size
                  << ", total_size=" << output_info[i].total_size
                  << ", inferred_elements=" << output_info[i].inferred_elements << std::endl;
        PrintTensorValueStats(frame_index, "output[" + std::to_string(i) + "]", outputs[i], output_info[i]);
    }

    std::cout << "[PALM][frame " << frame_index << "] output mapping: reg14=" << mapping.reg14
              << ", cls14=" << mapping.cls14
              << ", reg7=" << mapping.reg7
              << ", cls7=" << mapping.cls7
              << ", valid=" << (mapping.valid ? 1 : 0)
              << ", reason=" << mapping.reason
              << ", active_layout=" << OutputLayoutName(output_layout) << std::endl;

    std::cout << "[PALM][frame " << frame_index << "] detections=" << result.detections.size()
              << ", valid=" << (result.valid ? 1 : 0) << std::endl;
    for (size_t i = 0; i < result.detections.size(); i++) {
        const PalmDetection& det = result.detections[i];
        std::cout << "[PALM][frame " << frame_index << "] det[" << i
                  << "]: head=" << det.head_feature_size
                  << ", score=" << det.score
                  << ", model_box=(" << det.model_box[0] << "," << det.model_box[1]
                  << "," << det.model_box[2] << "," << det.model_box[3] << ")"
                  << ", pixel_box=(" << det.pixel_box[0] << "," << det.pixel_box[1]
                  << "," << det.pixel_box[2] << "," << det.pixel_box[3] << ")"
                  << std::endl;
        for (int kp = 0; kp < kPalmNumKeypoints; kp++) {
            const PalmKeypoint& point = det.keypoints[kp];
            std::cout << "[PALM][frame " << frame_index << "] det[" << i << "].keypoint[" << kp
                      << "]: model=(" << point.model_x << "," << point.model_y
                      << "), pixel=(" << point.pixel_x << "," << point.pixel_y
                      << "), norm=(" << point.x << "," << point.y << ")"
                      << std::endl;
        }
    }
}
