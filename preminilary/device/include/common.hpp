#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "smartsoc/ssne_api.h"

static const int kPalmFeature14 = 14;
static const int kPalmFeature7 = 7;
static const int kPalmNumAnchorsPerCell = 2;
static const int kPalmNumKeypoints = 2;
static const int kPalmValuesPerAnchor = 4 + kPalmNumKeypoints * 2;
static const int kPalmRegChannels = kPalmNumAnchorsPerCell * kPalmValuesPerAnchor;
static const int kPalmClsChannels = kPalmNumAnchorsPerCell;
static const int kPalmOutputCount = 4;
static const int kPalmMaxDetections = 2;
static const int kHandNumLandmarks = 21;
static const int kHandLandmarkValues = kHandNumLandmarks * 2;
static const int kHandOutputCount = 3;

static const float kPalmScoreThreshold = 0.50f;
static const float kPalmNmsIouThreshold = 0.30f;
static const float kPalmCrossHeadSuppressIou = 0.35f;

enum PalmOutputLayout {
    kPalmOutputLayoutNchw = 0,
    kPalmOutputLayoutHwc = 1,
};

struct PalmKeypoint {
    float model_x;
    float model_y;
    float x;
    float y;
    int pixel_x;
    int pixel_y;

    PalmKeypoint()
        : model_x(0.0f), model_y(0.0f), x(0.0f), y(0.0f), pixel_x(0), pixel_y(0) {}
};

struct PalmDetection {
    std::array<float, 4> model_box;
    std::array<float, 4> pixel_box;
    std::array<PalmKeypoint, kPalmNumKeypoints> keypoints;
    float score;
    int head_feature_size;

    PalmDetection()
        : model_box{{0.0f, 0.0f, 0.0f, 0.0f}},
          pixel_box{{0.0f, 0.0f, 0.0f, 0.0f}},
          score(0.0f),
          head_feature_size(0) {}
};

struct PalmResult {
    std::vector<PalmDetection> detections;
    bool valid;

    PalmResult() : valid(false) {}
    void Clear();
};

struct PalmPredictTiming {
    double preprocess_ms = 0.0;
    double preprocess_transform_ms = 0.0;
    double preprocess_manual_load_ms = 0.0;
    double input_load_ms = 0.0;
    double inference_ms = 0.0;
    double getoutput_ms = 0.0;
    double output_meta_ms = 0.0;
    double decode_ms = 0.0;
    double verbose_log_ms = 0.0;
    double accounted_ms = 0.0;
    bool success = false;
};

struct HandLandmark {
    float x;
    float y;
    int pixel_x;
    int pixel_y;

    HandLandmark() : x(0.0f), y(0.0f), pixel_x(0), pixel_y(0) {}
};

struct HandDetection {
    std::array<HandLandmark, kHandNumLandmarks> landmarks;
    float hand_flag_score;
    float handedness_score;
    bool has_hand_flag;
    bool has_handedness;
    bool valid;

    HandDetection()
        : hand_flag_score(0.0f),
          handedness_score(0.0f),
          has_hand_flag(false),
          has_handedness(false),
          valid(false) {}
};

struct HandResult {
    std::vector<HandDetection> detections;
    bool valid;

    HandResult() : valid(false) {}
    void Clear();
};

class IMAGEPROCESSOR {
public:
    void Initialize(std::array<int, 2>* in_img_shape);
    void GetImage(ssne_tensor_t* img_sensor);
    void Release();

    std::array<int, 2> img_shape;

private:
    uint8_t format_online = SSNE_Y_8;
};

class PALMDETECTOR {
public:
    PALMDETECTOR();

    void Initialize(const std::string& model_path,
                    const std::array<int, 2>& image_shape,
                    const std::array<int, 2>& input_shape,
                    bool use_ai_preprocess,
                    bool rotate_clockwise,
                    PalmOutputLayout output_layout);
    void Predict(ssne_tensor_t* img,
                 PalmResult* result,
                 uint32_t frame_index,
                 bool verbose_log,
                 PalmPredictTiming* timing = nullptr);
    void Release();

private:
    struct TensorDebugInfo {
        uint32_t width;
        uint32_t height;
        uint8_t dtype;
        uint8_t format;
        size_t mem_size;
        uint32_t total_size;
        size_t inferred_elements;
    };

    struct TensorValueStats {
        bool valid;
        size_t element_count;
        size_t finite_count;
        size_t nonzero_count;
        size_t near_half_count;
        double min_value;
        double max_value;
        double mean_value;
        double sample_first;
        double sample_center;
        double sample_last;

        TensorValueStats()
            : valid(false),
              element_count(0),
              finite_count(0),
              nonzero_count(0),
              near_half_count(0),
              min_value(0.0),
              max_value(0.0),
              mean_value(0.0),
              sample_first(0.0),
              sample_center(0.0),
              sample_last(0.0) {}
    };

    struct OutputMapping {
        int reg14;
        int cls14;
        int reg7;
        int cls7;
        bool valid;
        std::string reason;

        OutputMapping() : reg14(-1), cls14(-1), reg7(-1), cls7(-1), valid(false), reason() {}
    };

    struct Anchor {
        float cx;
        float cy;
        float w;
        float h;
    };

    struct Candidate {
        PalmDetection detection;
        int original_index;
    };

    static float Clamp01(float value);
    static TensorDebugInfo GetTensorDebugInfo(ssne_tensor_t tensor);
    static TensorValueStats GetTensorValueStats(ssne_tensor_t tensor, const TensorDebugInfo& info);
    static size_t InferElementCount(const TensorDebugInfo& info);
    static float ReadTensorValue(ssne_tensor_t tensor, const TensorDebugInfo& info, size_t index);
    static float IoU(const std::array<float, 4>& a, const std::array<float, 4>& b);
    static std::vector<int> NmsIndices(const std::vector<Candidate>& candidates, float iou_threshold);
    static bool IsSameIndexUsed(const std::vector<int>& values, int value);
    static size_t ExpectedRegElements(int feature_size);
    static size_t ExpectedClsElements(int feature_size);
    static const char* OutputLayoutName(PalmOutputLayout layout);

    OutputMapping MapOutputs(const TensorDebugInfo output_info[kPalmOutputCount]) const;
    bool PreprocessRotateResize(ssne_tensor_t camera_tensor, PalmPredictTiming* timing = nullptr);
    void ResizeBilinear(const uint8_t* src,
                        int src_width,
                        int src_height,
                        uint8_t* dst,
                        int dst_width,
                        int dst_height) const;
    void ResizeClockwiseRotatedBilinear(const uint8_t* src,
                                        int src_width,
                                        int src_height,
                                        uint8_t* dst,
                                        int dst_width,
                                        int dst_height) const;
    Anchor GetAnchor(int feature_size, int cell_x, int cell_y, int anchor_index) const;
    PalmKeypoint MapPoint(float model_x, float model_y) const;
    std::array<float, 4> MapBox(const std::array<float, 4>& model_box) const;
    size_t GetOutputIndex(int feature_size,
                          int channel_count,
                          int channel,
                          int cell_x,
                          int cell_y,
                          PalmOutputLayout layout) const;
    void DecodeHead(ssne_tensor_t reg_tensor,
                    const TensorDebugInfo& reg_info,
                    ssne_tensor_t cls_tensor,
                    const TensorDebugInfo& cls_info,
                    int feature_size,
                    PalmOutputLayout layout,
                    std::vector<Candidate>* candidates) const;
    void SelectDetections(const std::vector<Candidate>& candidates, PalmResult* result) const;
    void DecodeOutputs(const OutputMapping& mapping,
                       const TensorDebugInfo output_info[kPalmOutputCount],
                       PalmOutputLayout layout,
                       PalmResult* result) const;
    void PrintFrameLog(uint32_t frame_index,
                       ssne_tensor_t camera_tensor,
                       const TensorDebugInfo& manual_input_info,
                       const TensorDebugInfo& model_input_info,
                       const TensorDebugInfo output_info[kPalmOutputCount],
                       const OutputMapping& mapping,
                       const PalmResult& result) const;
    void PrintTensorValueStats(uint32_t frame_index,
                               const std::string& label,
                               ssne_tensor_t tensor,
                               const TensorDebugInfo& info) const;

    uint16_t model_id = 0;
    ssne_tensor_t manual_input;
    ssne_tensor_t inputs[1];
    ssne_tensor_t outputs[kPalmOutputCount];
    AiPreprocessPipe pipe_offline = nullptr;
    std::array<int, 2> image_shape = {720, 1280};
    std::array<int, 2> rotated_shape = {1280, 720};
    std::array<int, 2> input_shape = {224, 224};
    std::vector<uint8_t> manual_input_buffer;
    bool rotate_clockwise = true;
    PalmOutputLayout output_layout = kPalmOutputLayoutHwc;
    bool use_ai_preprocess = false;
    bool initialized = false;
};

class HANDLANDMARKER {
public:
    HANDLANDMARKER();

    void Initialize(const std::string& model_path,
                    const std::array<int, 2>& image_shape,
                    const std::array<int, 2>& input_shape);
    void Predict(ssne_tensor_t* img, const PalmResult& palm_result, HandResult* result);
    void Release();

private:
    struct TensorDebugInfo {
        uint32_t width;
        uint32_t height;
        uint8_t dtype;
        uint8_t format;
        size_t mem_size;
        uint32_t total_size;
        size_t inferred_elements;
    };

    struct RoiRect {
        float x_center;
        float y_center;
        float width;
        float height;
        float rotation;
        std::array<std::array<float, 2>, 4> corners;

        RoiRect()
            : x_center(0.0f),
              y_center(0.0f),
              width(0.0f),
              height(0.0f),
              rotation(0.0f),
              corners() {}
    };

    struct OutputMapping {
        int landmarks;
        int hand_flag;
        int handedness;
        bool valid;
        std::string reason;

        OutputMapping()
            : landmarks(-1), hand_flag(-1), handedness(-1), valid(false), reason() {}
    };

    static TensorDebugInfo GetTensorDebugInfo(ssne_tensor_t tensor);
    static size_t InferElementCount(const TensorDebugInfo& info);
    static float ReadTensorValue(ssne_tensor_t tensor, const TensorDebugInfo& info, size_t index);
    static float Clamp(float value, float low, float high);
    static float NormalizeRadians(float angle);
    static float NormalizeScore(float value);
    static const char* DTypeName(int dtype);
    static const char* FormatName(int format);

    OutputMapping MapOutputs(const TensorDebugInfo output_info[kHandOutputCount]) const;
    bool PreprocessRoi(ssne_tensor_t camera_tensor, const PalmDetection& palm_detection);
    bool PackRoiInputBuffer();
    RoiRect BuildRoiRect(const PalmDetection& palm_detection, int image_width, int image_height) const;
    uint8_t SampleBilinear(const uint8_t* src,
                           int src_width,
                           int src_height,
                           float x,
                           float y) const;
    void DecodeOutputs(const OutputMapping& mapping,
                       const TensorDebugInfo output_info[kHandOutputCount],
                       const RoiRect& rect,
                       HandDetection* detection) const;

    uint16_t model_id = 0;
    ssne_tensor_t input;
    ssne_tensor_t inputs[1];
    ssne_tensor_t outputs[kHandOutputCount];
    std::array<int, 2> image_shape = {720, 1280};
    std::array<int, 2> input_shape = {256, 256};
    std::vector<uint8_t> input_buffer;
    std::vector<uint8_t> roi_gray_buffer;
    RoiRect current_rect;
    uint8_t input_format = SSNE_Y_8;
    bool roi_debug_logged = false;
    bool input_pack_warning_logged = false;
    mutable bool decode_debug_logged = false;
    bool output_info_logged = false;
    bool initialized = false;
};
