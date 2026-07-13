#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

namespace slr_demo {

using Clock = std::chrono::steady_clock;

struct FrameTiming {
    double get_image_ms;
    double palm_total_ms;
    double palm_preprocess_ms;
    double palm_preprocess_transform_ms;
    double palm_preprocess_manual_load_ms;
    double palm_input_load_ms;
    double palm_inference_ms;
    double palm_getoutput_ms;
    double palm_output_meta_ms;
    double palm_decode_ms;
    double palm_verbose_log_ms;
    double palm_accounted_ms;
    double hand_total_ms;
    double draw_ms;
    double loop_ms;
    double process_ms;
};

double DurationMs(const Clock::time_point& begin, const Clock::time_point& end);

class PerformanceMonitor {
public:
    PerformanceMonitor(bool enabled,
                       double sensor_fps,
                       uint32_t report_interval_frames);

    void PrintConfig() const;
    void AddFrame(uint32_t frame_index, const FrameTiming& timing);

private:
    struct Stats {
        double avg;
        double p50;
        double p95;
        double max;
    };

    static double Clamp(double value, double low, double high);
    static Stats CalculateStats(std::vector<double> values);

    void PrintReport(uint32_t frame_index) const;
    void ClearWindow();

    bool enabled_;
    double sensor_fps_;
    double sensor_period_ms_;
    uint32_t report_interval_frames_;
    uint32_t total_frames_ = 0;
    Clock::time_point app_start_time_;
    std::vector<double> get_image_ms_;
    std::vector<double> palm_total_ms_;
    std::vector<double> palm_preprocess_ms_;
    std::vector<double> palm_preprocess_transform_ms_;
    std::vector<double> palm_preprocess_manual_load_ms_;
    std::vector<double> palm_input_load_ms_;
    std::vector<double> palm_inference_ms_;
    std::vector<double> palm_getoutput_ms_;
    std::vector<double> palm_output_meta_ms_;
    std::vector<double> palm_decode_ms_;
    std::vector<double> palm_verbose_log_ms_;
    std::vector<double> palm_accounted_ms_;
    std::vector<double> hand_total_ms_;
    std::vector<double> draw_ms_;
    std::vector<double> loop_ms_;
    std::vector<double> process_ms_;
    std::vector<double> instant_fps_;
};

}  // namespace slr_demo
