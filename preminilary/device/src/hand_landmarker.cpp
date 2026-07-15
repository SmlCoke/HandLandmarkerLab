#include "../include/common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

namespace {

const float kPi = 3.14159265358979323846f;
const float kTwoPi = 2.0f * kPi;
const float kHandRoiScaleX = 1.8f;
const float kHandRoiScaleY = 1.8f;
const float kHandRoiShiftX = 0.0f;
const float kHandRoiShiftY = -0.1f;
const uint8_t kHandModelInputFormat = SSNE_Y_8;

}  // namespace

void HandResult::Clear() {
    detections.clear();
    valid = false;
}

HANDLANDMARKER::HANDLANDMARKER() {
    input.data = nullptr;
    inputs[0].data = nullptr;
    for (int i = 0; i < kHandOutputCount; i++) {
        outputs[i].data = nullptr;
    }
}

void HANDLANDMARKER::Initialize(const std::string& model_path,
                                const std::array<int, 2>& in_image_shape,
                                const std::array<int, 2>& in_input_shape) {
    image_shape = in_image_shape;
    input_shape = in_input_shape;

    char* model_path_char = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_char, SSNE_STATIC_ALLOC);
    std::cout << "[HAND] Loaded model: " << model_path << ", model_id=" << model_id << std::endl;

    const int input_num = ssne_get_model_input_num(model_id);
    int input_dtype = -1;
    ssne_get_model_input_dtype(model_id, &input_dtype);

    int mean[3] = {0, 0, 0};
    int std_scale[3] = {0, 0, 0};
    int is_uint8 = 0;
    const int norm_ret = ssne_get_model_normalize_params(model_id, mean, std_scale, &is_uint8);
    std::cout << "[HAND] Model input_num=" << input_num
              << ", input_dtype=" << input_dtype << "(" << DTypeName(input_dtype) << ")"
              << ", normalize_ret=" << norm_ret
              << ", mean=(" << mean[0] << "," << mean[1] << "," << mean[2] << ")"
              << ", std=(" << std_scale[0] << "," << std_scale[1] << "," << std_scale[2] << ")"
              << ", is_uint8=" << is_uint8 << std::endl;

    input_format = kHandModelInputFormat;
    input = create_tensor(static_cast<uint32_t>(input_shape[0]),
                          static_cast<uint32_t>(input_shape[1]),
                          input_format,
                          SSNE_BUF_AI);
    inputs[0] = input;

    const TensorDebugInfo input_info = GetTensorDebugInfo(inputs[0]);
    input_buffer.resize(input_info.mem_size);
    roi_gray_buffer.resize(static_cast<size_t>(input_shape[0]) *
                           static_cast<size_t>(input_shape[1]));
    std::cout << "[HAND] ROI preprocessing: camera=" << image_shape[0] << "x" << image_shape[1]
              << " -> affine ROI gray8=" << input_shape[0] << "x" << input_shape[1]
              << " -> model tensor " << FormatName(input_format)
              << ", roi_scale=(" << kHandRoiScaleX << "," << kHandRoiScaleY << ")"
              << ", roi_shift=(" << kHandRoiShiftX << "," << kHandRoiShiftY << ")" << std::endl;
    std::cout << "[HAND] Model input tensor: width=" << input_info.width
              << ", height=" << input_info.height
              << ", dtype=" << static_cast<int>(input_info.dtype)
              << ", format=" << static_cast<int>(input_info.format)
              << ", mem_size=" << input_info.mem_size
              << ", total_size=" << input_info.total_size << std::endl;

    initialized = true;
}

void HANDLANDMARKER::Predict(ssne_tensor_t* img, const PalmResult& palm_result, HandResult* result) {
    result->Clear();
    if (!initialized) {
        std::cerr << "[HAND] Predict called before Initialize." << std::endl;
        return;
    }
    if (img == nullptr || !palm_result.valid || palm_result.detections.empty()) {
        return;
    }

    for (size_t palm_idx = 0; palm_idx < palm_result.detections.size(); palm_idx++) {
        if (!PreprocessRoi(*img, palm_result.detections[palm_idx])) {
            continue;
        }

        const int load_ret = load_tensor_buffer_ptr(inputs[0],
                                                    input_buffer.data(),
                                                    static_cast<int>(input_buffer.size()));
        if (load_ret != 0) {
            std::cerr << "[HAND] load_tensor_buffer_ptr failed, ret=" << load_ret << std::endl;
            continue;
        }

        const int infer_ret = ssne_inference(model_id, 1, inputs);
        if (infer_ret != 0) {
            std::cerr << "[HAND] ssne_inference failed, ret=" << infer_ret << std::endl;
            continue;
        }

        const int output_ret = ssne_getoutput(model_id, kHandOutputCount, outputs);
        if (output_ret != 0) {
            std::cerr << "[HAND] ssne_getoutput failed, ret=" << output_ret << std::endl;
            continue;
        }

        TensorDebugInfo output_info[kHandOutputCount];
        for (int i = 0; i < kHandOutputCount; i++) {
            output_info[i] = GetTensorDebugInfo(outputs[i]);
        }

        const OutputMapping mapping = MapOutputs(output_info);
        if (!output_info_logged) {
            std::cout << "[HAND] Output tensors:";
            for (int i = 0; i < kHandOutputCount; i++) {
                std::cout << " out" << i
                          << "(w=" << output_info[i].width
                          << ",h=" << output_info[i].height
                          << ",dtype=" << static_cast<int>(output_info[i].dtype)
                          << ",elements=" << output_info[i].inferred_elements
                          << ")";
            }
            std::cout << " mapping(landmarks=" << mapping.landmarks
                      << ",hand_flag=" << mapping.hand_flag
                      << ",handedness=" << mapping.handedness
                      << ")" << std::endl;
            output_info_logged = true;
        }
        if (!mapping.valid) {
            std::cerr << "[HAND] Output mapping failed: " << mapping.reason << std::endl;
            continue;
        }

        HandDetection detection;
        DecodeOutputs(mapping, output_info, current_rect, &detection);
        if (detection.valid) {
            result->detections.push_back(detection);
        }
    }

    result->valid = !result->detections.empty();
}

void HANDLANDMARKER::Release() {
    if (input.data != nullptr) {
        release_tensor(input);
        input.data = nullptr;
        inputs[0].data = nullptr;
    }

    for (int i = 0; i < kHandOutputCount; i++) {
        if (outputs[i].data != nullptr) {
            release_tensor(outputs[i]);
            outputs[i].data = nullptr;
        }
    }

    initialized = false;
}

HANDLANDMARKER::TensorDebugInfo HANDLANDMARKER::GetTensorDebugInfo(ssne_tensor_t tensor) {
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

size_t HANDLANDMARKER::InferElementCount(const TensorDebugInfo& info) {
    if (info.dtype == SSNE_FLOAT32) {
        return info.mem_size / sizeof(float);
    }
    if (info.dtype == SSNE_UINT8 || info.dtype == SSNE_INT8) {
        return info.mem_size;
    }
    return info.total_size;
}

float HANDLANDMARKER::ReadTensorValue(ssne_tensor_t tensor,
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

float HANDLANDMARKER::Clamp(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

float HANDLANDMARKER::NormalizeRadians(float angle) {
    return angle - kTwoPi * std::floor((angle + kPi) / kTwoPi);
}

const char* HANDLANDMARKER::DTypeName(int dtype) {
    if (dtype == SSNE_UINT8) {
        return "UINT8";
    }
    if (dtype == SSNE_INT8) {
        return "INT8";
    }
    if (dtype == SSNE_FLOAT32) {
        return "FLOAT32";
    }
    return "UNKNOWN";
}

const char* HANDLANDMARKER::FormatName(int format) {
    if (format == SSNE_BYTES) {
        return "SSNE_BYTES";
    }
    if (format == SSNE_YUV422_20) {
        return "SSNE_YUV422_20";
    }
    if (format == SSNE_YUV422_16) {
        return "SSNE_YUV422_16";
    }
    if (format == SSNE_Y_10) {
        return "SSNE_Y_10";
    }
    if (format == SSNE_Y_8) {
        return "SSNE_Y_8";
    }
    if (format == SSNE_RGB) {
        return "SSNE_RGB";
    }
    if (format == SSNE_BGR) {
        return "SSNE_BGR";
    }
    return "UNKNOWN";
}

HANDLANDMARKER::OutputMapping HANDLANDMARKER::MapOutputs(
    const TensorDebugInfo output_info[kHandOutputCount]) const {
    OutputMapping mapping;
    std::vector<int> scalar_outputs;

    for (int i = 0; i < kHandOutputCount; i++) {
        if (output_info[i].dtype != SSNE_FLOAT32) {
            mapping.reason = "all Hand outputs must be FLOAT32";
            return mapping;
        }
        if (output_info[i].inferred_elements == kHandLandmarkValues && mapping.landmarks < 0) {
            mapping.landmarks = i;
        } else if (output_info[i].inferred_elements == 1) {
            scalar_outputs.push_back(i);
        }
    }

    for (size_t i = 0; i < scalar_outputs.size(); i++) {
        if (mapping.hand_flag < 0) {
            mapping.hand_flag = scalar_outputs[i];
        } else if (mapping.handedness < 0) {
            mapping.handedness = scalar_outputs[i];
        }
    }

    mapping.valid = mapping.landmarks >= 0 && mapping.hand_flag >= 0 && mapping.handedness >= 0 &&
                    scalar_outputs.size() == 2;
    mapping.reason = mapping.valid
                         ? "matched strict FLOAT32 outputs with element counts 42,1,1"
                         : "Hand outputs must contain exactly one 42-value tensor and two scalar tensors";
    return mapping;
}

bool HANDLANDMARKER::PreprocessRoi(ssne_tensor_t camera_tensor,
                                   const PalmDetection& palm_detection) {
    const uint8_t* camera_data = reinterpret_cast<const uint8_t*>(get_data(camera_tensor));
    if (camera_data == nullptr) {
        std::cerr << "[HAND] Camera tensor has null data." << std::endl;
        return false;
    }

    const TensorDebugInfo camera_info = GetTensorDebugInfo(camera_tensor);
    const int src_width = camera_info.width > 0 ? static_cast<int>(camera_info.width) : image_shape[0];
    const int src_height = camera_info.height > 0 ? static_cast<int>(camera_info.height) : image_shape[1];
    if (camera_info.format != SSNE_Y_8) {
        std::cerr << "[HAND] Warning: expected SSNE_Y_8 camera tensor, got format="
                  << static_cast<int>(camera_info.format) << std::endl;
    }

    current_rect = BuildRoiRect(palm_detection, src_width, src_height);
    if (current_rect.width < 2.0f || current_rect.height < 2.0f) {
        return false;
    }

    const std::array<float, 2>& top_left = current_rect.corners[0];
    const std::array<float, 2>& top_right = current_rect.corners[1];
    const std::array<float, 2>& bottom_left = current_rect.corners[3];

    const float denom_x = static_cast<float>(std::max(1, input_shape[0] - 1));
    const float denom_y = static_cast<float>(std::max(1, input_shape[1] - 1));
    for (int y = 0; y < input_shape[1]; y++) {
        const float ty = static_cast<float>(y) / denom_y;
        for (int x = 0; x < input_shape[0]; x++) {
            const float tx = static_cast<float>(x) / denom_x;
            const float src_x = top_left[0] +
                                tx * (top_right[0] - top_left[0]) +
                                ty * (bottom_left[0] - top_left[0]);
            const float src_y = top_left[1] +
                                tx * (top_right[1] - top_left[1]) +
                                ty * (bottom_left[1] - top_left[1]);
            roi_gray_buffer[static_cast<size_t>(y) * static_cast<size_t>(input_shape[0]) +
                            static_cast<size_t>(x)] =
                SampleBilinear(camera_data, src_width, src_height, src_x, src_y);
        }
    }

    if (!roi_debug_logged) {
        uint8_t min_value = 255;
        uint8_t max_value = 0;
        uint64_t sum_value = 0;
        for (size_t i = 0; i < roi_gray_buffer.size(); i++) {
            min_value = std::min(min_value, roi_gray_buffer[i]);
            max_value = std::max(max_value, roi_gray_buffer[i]);
            sum_value += roi_gray_buffer[i];
        }
        const double mean_value = roi_gray_buffer.empty()
                                      ? 0.0
                                      : static_cast<double>(sum_value) /
                                            static_cast<double>(roi_gray_buffer.size());
        std::cout << "[HAND][debug] roi_rect center=(" << current_rect.x_center
                  << "," << current_rect.y_center << ")"
                  << ", size=(" << current_rect.width << "," << current_rect.height << ")"
                  << ", rotation_rad=" << current_rect.rotation
                  << ", corners=("
                  << current_rect.corners[0][0] << "," << current_rect.corners[0][1] << ";"
                  << current_rect.corners[1][0] << "," << current_rect.corners[1][1] << ";"
                  << current_rect.corners[2][0] << "," << current_rect.corners[2][1] << ";"
                  << current_rect.corners[3][0] << "," << current_rect.corners[3][1] << ")"
                  << std::endl;
        std::cout << "[HAND][debug] roi_gray8_stats min=" << static_cast<int>(min_value)
                  << ", max=" << static_cast<int>(max_value)
                  << ", mean=" << mean_value
                  << ", input_format=" << FormatName(input_format)
                  << ", input_bytes=" << input_buffer.size()
                  << std::endl;
        roi_debug_logged = true;
    }

    return PackRoiInputBuffer();
}

bool HANDLANDMARKER::PackRoiInputBuffer() {
    const size_t pixel_count = static_cast<size_t>(input_shape[0]) *
                               static_cast<size_t>(input_shape[1]);
    if (roi_gray_buffer.size() < pixel_count || input_buffer.empty()) {
        return false;
    }

    if (input_format == SSNE_Y_8) {
        if (input_buffer.size() < pixel_count) {
            return false;
        }
        std::copy(roi_gray_buffer.begin(), roi_gray_buffer.begin() + pixel_count, input_buffer.begin());
        return true;
    }

    if (!input_pack_warning_logged) {
        std::cerr << "[HAND] Unsupported input tensor packing: format=" << FormatName(input_format)
                  << ", bytes=" << input_buffer.size()
                  << ", pixels=" << pixel_count << std::endl;
        input_pack_warning_logged = true;
    }
    return false;
}

HANDLANDMARKER::RoiRect HANDLANDMARKER::BuildRoiRect(const PalmDetection& palm_detection,
                                                     int image_width,
                                                     int image_height) const {
    RoiRect rect;

    const float x1 = Clamp(std::min(palm_detection.pixel_box[0], palm_detection.pixel_box[2]),
                           0.0f,
                           static_cast<float>(image_width - 1));
    const float y1 = Clamp(std::min(palm_detection.pixel_box[1], palm_detection.pixel_box[3]),
                           0.0f,
                           static_cast<float>(image_height - 1));
    const float x2 = Clamp(std::max(palm_detection.pixel_box[0], palm_detection.pixel_box[2]),
                           0.0f,
                           static_cast<float>(image_width - 1));
    const float y2 = Clamp(std::max(palm_detection.pixel_box[1], palm_detection.pixel_box[3]),
                           0.0f,
                           static_cast<float>(image_height - 1));

    const float raw_width = std::max(1.0f, x2 - x1);
    const float raw_height = std::max(1.0f, y2 - y1);
    float center_x = 0.5f * (x1 + x2);
    float center_y = 0.5f * (y1 + y2);

    const PalmKeypoint& wrist = palm_detection.keypoints[0];
    const PalmKeypoint& middle = palm_detection.keypoints[1];
    const float dx = static_cast<float>(middle.pixel_x - wrist.pixel_x);
    const float dy = static_cast<float>(middle.pixel_y - wrist.pixel_y);
    const float rotation = NormalizeRadians((kPi * 0.5f) - std::atan2(-dy, dx));
    const float cos_r = std::cos(rotation);
    const float sin_r = std::sin(rotation);

    center_x += raw_width * kHandRoiShiftX * cos_r - raw_height * kHandRoiShiftY * sin_r;
    center_y += raw_width * kHandRoiShiftX * sin_r + raw_height * kHandRoiShiftY * cos_r;

    const float long_side = std::max(raw_width, raw_height);
    const float roi_width = long_side * kHandRoiScaleX;
    const float roi_height = long_side * kHandRoiScaleY;

    const float vx_x = cos_r * roi_width * 0.5f;
    const float vx_y = sin_r * roi_width * 0.5f;
    const float vy_x = -sin_r * roi_height * 0.5f;
    const float vy_y = cos_r * roi_height * 0.5f;

    rect.x_center = center_x;
    rect.y_center = center_y;
    rect.width = roi_width;
    rect.height = roi_height;
    rect.rotation = rotation;
    rect.corners[0] = {{center_x - vx_x - vy_x, center_y - vx_y - vy_y}};
    rect.corners[1] = {{center_x + vx_x - vy_x, center_y + vx_y - vy_y}};
    rect.corners[2] = {{center_x + vx_x + vy_x, center_y + vx_y + vy_y}};
    rect.corners[3] = {{center_x - vx_x + vy_x, center_y - vx_y + vy_y}};
    return rect;
}

uint8_t HANDLANDMARKER::SampleBilinear(const uint8_t* src,
                                       int src_width,
                                       int src_height,
                                       float x,
                                       float y) const {
    if (x < -1.0f || y < -1.0f ||
        x > static_cast<float>(src_width) ||
        y > static_cast<float>(src_height)) {
        return 0;
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float wx = x - static_cast<float>(x0);
    const float wy = y - static_cast<float>(y0);

    const auto read_pixel = [src, src_width, src_height](int px, int py) -> float {
        if (px < 0 || py < 0 || px >= src_width || py >= src_height) {
            return 0.0f;
        }
        return static_cast<float>(src[py * src_width + px]);
    };

    const float v00 = read_pixel(x0, y0);
    const float v01 = read_pixel(x1, y0);
    const float v10 = read_pixel(x0, y1);
    const float v11 = read_pixel(x1, y1);
    const float top = v00 * (1.0f - wx) + v01 * wx;
    const float bottom = v10 * (1.0f - wx) + v11 * wx;
    const float value = top * (1.0f - wy) + bottom * wy;
    return static_cast<uint8_t>(Clamp(std::round(value), 0.0f, 255.0f));
}

void HANDLANDMARKER::DecodeOutputs(const OutputMapping& mapping,
                                   const TensorDebugInfo output_info[kHandOutputCount],
                                   const RoiRect& rect,
                                   HandDetection* detection) const {
    detection->valid = false;

    if (mapping.hand_flag >= 0) {
        detection->hand_flag_score = Clamp(
            ReadTensorValue(outputs[mapping.hand_flag],
                            output_info[mapping.hand_flag],
                            0),
            0.0f,
            1.0f);
        detection->has_hand_flag = true;
    }

    if (mapping.handedness >= 0) {
        detection->handedness_score = Clamp(
            ReadTensorValue(outputs[mapping.handedness],
                            output_info[mapping.handedness],
                            0),
            0.0f,
            1.0f);
        detection->has_handedness = true;
    }

    const TensorDebugInfo& landmark_info = output_info[mapping.landmarks];
    float raw_values[kHandLandmarkValues];
    float max_abs = 0.0f;
    float min_raw = std::numeric_limits<float>::max();
    float max_raw = std::numeric_limits<float>::lowest();
    for (int i = 0; i < kHandLandmarkValues; i++) {
        raw_values[i] = ReadTensorValue(outputs[mapping.landmarks], landmark_info, static_cast<size_t>(i));
        max_abs = std::max(max_abs, std::fabs(raw_values[i]));
        min_raw = std::min(min_raw, raw_values[i]);
        max_raw = std::max(max_raw, raw_values[i]);
    }

    const std::array<float, 2>& top_left = rect.corners[0];
    const std::array<float, 2>& top_right = rect.corners[1];
    const std::array<float, 2>& bottom_left = rect.corners[3];
    float min_px = std::numeric_limits<float>::max();
    float max_px = std::numeric_limits<float>::lowest();
    float min_py = std::numeric_limits<float>::max();
    float max_py = std::numeric_limits<float>::lowest();

    for (int i = 0; i < kHandNumLandmarks; i++) {
        const float x = raw_values[i * 2];
        const float y = raw_values[i * 2 + 1];
        const float px = top_left[0] +
                         x * (top_right[0] - top_left[0]) +
                         y * (bottom_left[0] - top_left[0]);
        const float py = top_left[1] +
                         x * (top_right[1] - top_left[1]) +
                         y * (bottom_left[1] - top_left[1]);

        detection->landmarks[i].x = px;
        detection->landmarks[i].y = py;
        detection->landmarks[i].pixel_x = static_cast<int>(std::round(px));
        detection->landmarks[i].pixel_y = static_cast<int>(std::round(py));
        min_px = std::min(min_px, px);
        max_px = std::max(max_px, px);
        min_py = std::min(min_py, py);
        max_py = std::max(max_py, py);
    }

    if (!decode_debug_logged) {
        std::cout << "[HAND][debug] landmark_raw min=" << min_raw
                  << ", max=" << max_raw
                  << ", max_abs=" << max_abs
                  << ", hand_flag=" << (detection->has_hand_flag ? detection->hand_flag_score : -1.0f)
                  << ", handedness=" << (detection->has_handedness ? detection->handedness_score : -1.0f)
                  << std::endl;
        std::cout << "[HAND][debug] projected_landmark_bbox=(" << min_px << "," << min_py
                  << "," << max_px << "," << max_py << ")"
                  << ", wrist=(" << detection->landmarks[0].x << "," << detection->landmarks[0].y << ")"
                  << ", middle_mcp=(" << detection->landmarks[9].x << "," << detection->landmarks[9].y << ")"
                  << ", middle_tip=(" << detection->landmarks[12].x << "," << detection->landmarks[12].y << ")"
                  << std::endl;
        decode_debug_logged = true;
    }

    detection->valid = true;
}
