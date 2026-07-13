#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>

#include "include/performance_monitor.hpp"
#include "include/utils.hpp"

namespace {

using Clock = slr_demo::Clock;
using slr_demo::DurationMs;
using slr_demo::FrameTiming;
using slr_demo::PerformanceMonitor;

bool g_exit_flag = false;
std::mutex g_exit_mutex;

void keyboard_listener() {
    std::string input;
    std::cout << "[PALM_DEMO] Keyboard listener started. Input 'q' to exit." << std::endl;

    while (std::cin >> input) {
        if (input == "q" || input == "Q") {
            std::lock_guard<std::mutex> lock(g_exit_mutex);
            g_exit_flag = true;
            std::cout << "[PALM_DEMO] Exit command received." << std::endl;
            return;
        }
        std::cout << "[PALM_DEMO] Unknown command. Input 'q' to exit." << std::endl;
    }
}

bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_exit_mutex);
    return g_exit_flag;
}

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [--kInferInterval N] [--enable_hand]" << std::endl;
    std::cerr << "       " << program_name << " [--kInferInterval=N] [--enable_hand]" << std::endl;
}

bool parse_uint32(const std::string& text, uint32_t* value) {
    if (value == nullptr || text.empty()) {
        return false;
    }
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] < '0' || text[i] > '9') {
            return false;
        }
    }

    errno = 0;
    char* end_ptr = nullptr;
    const unsigned long parsed = std::strtoul(text.c_str(), &end_ptr, 10);
    if (errno != 0 || end_ptr == text.c_str() || *end_ptr != '\0' ||
        parsed > static_cast<unsigned long>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }

    *value = static_cast<uint32_t>(parsed);
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::array<int, 2> image_shape = {720, 1280};
    const std::array<int, 2> palm_input_shape = {224, 224};
    const std::array<int, 2> hand_input_shape = {256, 256};
    const std::string palm_model_path = "/app_demo/app_assets/models/palm.m1model";
    const std::string hand_model_path = "/app_demo/app_assets/models/hand.m1model";
    const bool verbose = false;
    uint32_t kInferInterval = 1;
    bool enable_hand = false;

    for (int arg_idx = 1; arg_idx < argc; arg_idx++) {
        const std::string arg = argv[arg_idx];
        if (arg == "--enable_hand") {
            enable_hand = true;
        } else if (arg == "--kInferInterval") {
            if (arg_idx + 1 >= argc || !parse_uint32(argv[arg_idx + 1], &kInferInterval)) {
                std::cerr << "[PALM_DEMO] Invalid --kInferInterval value." << std::endl;
                print_usage(argv[0]);
                return -1;
            }
            arg_idx += 1;
        } else if (arg.find("--kInferInterval=") == 0) {
            const std::string value = arg.substr(std::string("--kInferInterval=").size());
            if (!parse_uint32(value, &kInferInterval)) {
                std::cerr << "[PALM_DEMO] Invalid --kInferInterval value." << std::endl;
                print_usage(argv[0]);
                return -1;
            }
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "[PALM_DEMO] Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return -1;
        }
    }

    const uint32_t infer_interval = kInferInterval == 0 ? 1 : kInferInterval;
    const uint32_t verbose_multiple = (50 + infer_interval / 2) / infer_interval;
    const uint32_t verbose_interval =
        (verbose_multiple == 0 ? 1 : verbose_multiple) * infer_interval;
    const bool use_ai_preprocess = false;
    const bool rotate_clockwise = true;
    const PalmOutputLayout output_layout = kPalmOutputLayoutHwc;

    const bool perf_log_enabled = true;
    const double sensor_fps_cfg = 90.0;
    const uint32_t perf_report_interval_frames = 120;

    std::cout << "[PALM_DEMO] Starting palm detection demo." << std::endl;
    std::cout << "[PALM_DEMO] Inference interval=" << infer_interval
              << ", verbose_interval=" << verbose_interval
              << ", enable_hand=" << (enable_hand ? 1 : 0) << std::endl;

    if (ssne_initial()) {
        std::cerr << "[PALM_DEMO] SSNE initialization failed." << std::endl;
        return -1;
    }

    IMAGEPROCESSOR processor;
    std::array<int, 2> mutable_image_shape = image_shape;
    processor.Initialize(&mutable_image_shape);

    PALMDETECTOR palm_model;
    palm_model.Initialize(palm_model_path,
                          image_shape,
                          palm_input_shape,
                          use_ai_preprocess,
                          rotate_clockwise,
                          output_layout);

    HANDLANDMARKER hand_model;
    if (enable_hand) {
        hand_model.Initialize(hand_model_path, image_shape, hand_input_shape);
    } else {
        std::cout << "[HAND] Disabled. Pass --enable_hand to run the hand landmarker." << std::endl;
    }

    VISUALIZER visualizer;
    visualizer.Initialize(mutable_image_shape);

    std::cout << "[PALM_DEMO] Waiting 0.2s for pipeline stabilization." << std::endl;
    usleep(200000);

    PerformanceMonitor perf_monitor(perf_log_enabled,
                                    sensor_fps_cfg,
                                    perf_report_interval_frames);
    perf_monitor.PrintConfig();

    ssne_tensor_t image_tensor;
    PalmResult palm_result;
    HandResult hand_result;
    uint32_t frame_index = 0;
    std::thread listener_thread(keyboard_listener);

    while (!check_exit_flag()) {
        const Clock::time_point loop_begin = Clock::now();

        const Clock::time_point get_begin = Clock::now();
        processor.GetImage(&image_tensor);
        const Clock::time_point get_end = Clock::now();

        PalmPredictTiming palm_timing;
        Clock::time_point palm_begin = get_end;
        Clock::time_point palm_end = get_end;
        Clock::time_point hand_begin = get_end;
        Clock::time_point hand_end = get_end;
        if (frame_index % infer_interval == 0) {
            const bool verbose_log = verbose && (frame_index % verbose_interval == 0);
            palm_begin = Clock::now();
            palm_model.Predict(&image_tensor, &palm_result, frame_index, verbose_log, &palm_timing);
            palm_end = Clock::now();

            if (enable_hand) {
                hand_begin = Clock::now();
                hand_model.Predict(&image_tensor, palm_result, &hand_result);
                hand_end = Clock::now();
            } else {
                hand_result.Clear();
            }
        }

        const Clock::time_point draw_begin = Clock::now();
        visualizer.DrawDetections(palm_result, hand_result);
        const Clock::time_point draw_end = Clock::now();

        const FrameTiming timing = {
            DurationMs(get_begin, get_end),
            DurationMs(palm_begin, palm_end),
            palm_timing.preprocess_ms,
            palm_timing.preprocess_transform_ms,
            palm_timing.preprocess_manual_load_ms,
            palm_timing.input_load_ms,
            palm_timing.inference_ms,
            palm_timing.getoutput_ms,
            palm_timing.output_meta_ms,
            palm_timing.decode_ms,
            palm_timing.verbose_log_ms,
            palm_timing.accounted_ms,
            DurationMs(hand_begin, hand_end),
            DurationMs(draw_begin, draw_end),
            DurationMs(loop_begin, draw_end),
            DurationMs(get_end, draw_end),
        };
        perf_monitor.AddFrame(frame_index, timing);

        frame_index += 1;
    }

    if (listener_thread.joinable()) {
        listener_thread.join();
    }

    visualizer.Release();
    if (enable_hand) {
        hand_model.Release();
    }
    palm_model.Release();
    processor.Release();

    if (ssne_release()) {
        std::cerr << "[PALM_DEMO] SSNE release failed." << std::endl;
        return -1;
    }

    std::cout << "[PALM_DEMO] Palm detection demo stopped." << std::endl;
    return 0;
}
