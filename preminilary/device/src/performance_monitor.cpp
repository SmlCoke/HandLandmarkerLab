#include "../include/performance_monitor.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace slr_demo {

double DurationMs(const Clock::time_point& begin, const Clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - begin).count();
}

PerformanceMonitor::PerformanceMonitor(bool enabled,
                                       double sensor_fps,
                                       uint32_t report_interval_frames)
    : enabled_(enabled),
      sensor_fps_(sensor_fps > 0.0 ? sensor_fps : 90.0),
      sensor_period_ms_(1000.0 / (sensor_fps > 0.0 ? sensor_fps : 90.0)),
      report_interval_frames_(report_interval_frames > 0 ? report_interval_frames : 120),
      app_start_time_(Clock::now()) {}

void PerformanceMonitor::PrintConfig() const {
    if (!enabled_) {
        return;
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[PERF][config] enabled=1"
              << " sensor_fps_cfg=" << sensor_fps_
              << " sensor_period_ms=" << sensor_period_ms_
              << " report_interval_frames=" << report_interval_frames_
              << " note=\"e2e_loop includes GetImage wait; e2e_process starts after GetImage returns\""
              << std::endl;
}

void PerformanceMonitor::AddFrame(uint32_t frame_index, const FrameTiming& timing) {
    if (!enabled_) {
        return;
    }

    total_frames_ += 1;
    get_image_ms_.push_back(timing.get_image_ms);
    palm_total_ms_.push_back(timing.palm_total_ms);
    palm_preprocess_ms_.push_back(timing.palm_preprocess_ms);
    palm_preprocess_transform_ms_.push_back(timing.palm_preprocess_transform_ms);
    palm_preprocess_manual_load_ms_.push_back(timing.palm_preprocess_manual_load_ms);
    palm_input_load_ms_.push_back(timing.palm_input_load_ms);
    palm_inference_ms_.push_back(timing.palm_inference_ms);
    palm_getoutput_ms_.push_back(timing.palm_getoutput_ms);
    palm_output_meta_ms_.push_back(timing.palm_output_meta_ms);
    palm_decode_ms_.push_back(timing.palm_decode_ms);
    palm_verbose_log_ms_.push_back(timing.palm_verbose_log_ms);
    palm_accounted_ms_.push_back(timing.palm_accounted_ms);
    hand_total_ms_.push_back(timing.hand_total_ms);
    draw_ms_.push_back(timing.draw_ms);
    loop_ms_.push_back(timing.loop_ms);
    process_ms_.push_back(timing.process_ms);
    if (timing.loop_ms > 0.0) {
        instant_fps_.push_back(1000.0 / timing.loop_ms);
    }

    if (total_frames_ % report_interval_frames_ == 0) {
        PrintReport(frame_index);
        ClearWindow();
    }
}

double PerformanceMonitor::Clamp(double value, double low, double high) {
    return std::max(low, std::min(value, high));
}

PerformanceMonitor::Stats PerformanceMonitor::CalculateStats(std::vector<double> values) {
    Stats stats = {0.0, 0.0, 0.0, 0.0};
    if (values.empty()) {
        return stats;
    }

    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    const size_t p50_index = (n - 1) / 2;
    size_t p95_index = static_cast<size_t>(std::ceil(static_cast<double>(n) * 0.95));
    if (p95_index == 0) {
        p95_index = 1;
    }
    p95_index = std::min(p95_index - 1, n - 1);

    stats.avg = sum / static_cast<double>(n);
    stats.p50 = values[p50_index];
    stats.p95 = values[p95_index];
    stats.max = values[n - 1];
    return stats;
}

void PerformanceMonitor::PrintReport(uint32_t frame_index) const {
    const Stats get_stats = CalculateStats(get_image_ms_);
    const Stats palm_total_stats = CalculateStats(palm_total_ms_);
    const Stats palm_preprocess_stats = CalculateStats(palm_preprocess_ms_);
    const Stats palm_preprocess_transform_stats = CalculateStats(palm_preprocess_transform_ms_);
    const Stats palm_preprocess_manual_load_stats = CalculateStats(palm_preprocess_manual_load_ms_);
    const Stats palm_input_load_stats = CalculateStats(palm_input_load_ms_);
    const Stats palm_inference_stats = CalculateStats(palm_inference_ms_);
    const Stats palm_getoutput_stats = CalculateStats(palm_getoutput_ms_);
    const Stats palm_output_meta_stats = CalculateStats(palm_output_meta_ms_);
    const Stats palm_decode_stats = CalculateStats(palm_decode_ms_);
    const Stats palm_verbose_log_stats = CalculateStats(palm_verbose_log_ms_);
    const Stats palm_accounted_stats = CalculateStats(palm_accounted_ms_);
    const Stats hand_total_stats = CalculateStats(hand_total_ms_);
    const Stats draw_stats = CalculateStats(draw_ms_);
    const Stats loop_stats = CalculateStats(loop_ms_);
    const Stats process_stats = CalculateStats(process_ms_);
    const Stats fps_stats = CalculateStats(instant_fps_);

    const double elapsed_s =
        std::chrono::duration_cast<std::chrono::duration<double>>(Clock::now() - app_start_time_).count();
    const double app_fps_total =
        elapsed_s > 0.0 ? static_cast<double>(total_frames_) / elapsed_s : 0.0;
    const double window_ms = std::accumulate(loop_ms_.begin(), loop_ms_.end(), 0.0);
    const double app_fps_window =
        window_ms > 0.0 ? static_cast<double>(loop_ms_.size()) * 1000.0 / window_ms : 0.0;
    const double realtime_ratio = app_fps_window / sensor_fps_;
    const int realtime_score =
        static_cast<int>(std::floor(10.0 * Clamp(realtime_ratio, 0.0, 1.0)));

    const double expected_total_frames = elapsed_s * sensor_fps_;
    const double drop_rate_est =
        expected_total_frames > 0.0
            ? 100.0 * Clamp((expected_total_frames - static_cast<double>(total_frames_)) /
                                expected_total_frames,
                            0.0,
                            1.0)
            : 0.0;
    const double loop_jitter_pct =
        loop_stats.avg > 0.0 ? 100.0 * std::fabs(loop_stats.p95 - loop_stats.avg) / loop_stats.avg : 0.0;
    const double fps_jitter_pct =
        fps_stats.avg > 0.0 ? 100.0 * std::fabs(fps_stats.p95 - fps_stats.avg) / fps_stats.avg : 0.0;

    const double loop_p95_t = loop_stats.p95 / sensor_period_ms_;
    const double process_p95_t = process_stats.p95 / sensor_period_ms_;
    int latency_score = static_cast<int>(std::floor(11.0 - loop_p95_t));
    latency_score = static_cast<int>(Clamp(static_cast<double>(latency_score), 0.0, 10.0));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[PERF][summary]"
              << " frame=" << frame_index
              << " total_frames=" << total_frames_
              << " window_frames=" << loop_ms_.size()
              << " elapsed_s=" << elapsed_s
              << " sensor_fps_cfg=" << sensor_fps_
              << " app_fps_total=" << app_fps_total
              << " app_fps_window=" << app_fps_window
              << " R=" << realtime_ratio
              << " realtime_score_est=" << realtime_score
              << " drop_rate_est_pct=" << drop_rate_est
              << std::endl;

    std::cout << "[PERF][stage_ms]"
              << " get_image_avg=" << get_stats.avg << " get_image_p95=" << get_stats.p95
              << " palm_total_avg=" << palm_total_stats.avg << " palm_total_p95=" << palm_total_stats.p95
              << " hand_total_avg=" << hand_total_stats.avg << " hand_total_p95=" << hand_total_stats.p95
              << " draw_avg=" << draw_stats.avg << " draw_p95=" << draw_stats.p95
              << " process_avg=" << process_stats.avg << " process_p95=" << process_stats.p95
              << " loop_avg=" << loop_stats.avg << " loop_p95=" << loop_stats.p95
              << " loop_max=" << loop_stats.max
              << std::endl;

    std::cout << "[PERF][palm_detail_ms]"
              << " palm_preprocess_avg=" << palm_preprocess_stats.avg
              << " palm_preprocess_p95=" << palm_preprocess_stats.p95
              << " palm_preprocess_transform_avg=" << palm_preprocess_transform_stats.avg
              << " palm_preprocess_transform_p95=" << palm_preprocess_transform_stats.p95
              << " palm_preprocess_manual_load_avg=" << palm_preprocess_manual_load_stats.avg
              << " palm_preprocess_manual_load_p95=" << palm_preprocess_manual_load_stats.p95
              << " palm_input_load_avg=" << palm_input_load_stats.avg
              << " palm_input_load_p95=" << palm_input_load_stats.p95
              << " palm_inference_avg=" << palm_inference_stats.avg
              << " palm_inference_p95=" << palm_inference_stats.p95
              << " palm_getoutput_avg=" << palm_getoutput_stats.avg
              << " palm_getoutput_p95=" << palm_getoutput_stats.p95
              << " palm_output_meta_avg=" << palm_output_meta_stats.avg
              << " palm_output_meta_p95=" << palm_output_meta_stats.p95
              << " palm_decode_avg=" << palm_decode_stats.avg
              << " palm_decode_p95=" << palm_decode_stats.p95
              << " palm_verbose_log_avg=" << palm_verbose_log_stats.avg
              << " palm_verbose_log_p95=" << palm_verbose_log_stats.p95
              << " palm_accounted_avg=" << palm_accounted_stats.avg
              << " palm_accounted_p95=" << palm_accounted_stats.p95
              << std::endl;

    std::cout << "[PERF][latency]"
              << " sensor_period_ms=" << sensor_period_ms_
              << " e2e_loop_p95_ms=" << loop_stats.p95
              << " e2e_loop_p95_T=" << loop_p95_t
              << " e2e_process_p95_ms=" << process_stats.p95
              << " e2e_process_p95_T=" << process_p95_t
              << " latency_score_est_by_loop=" << latency_score
              << std::endl;

    std::cout << "[PERF][jitter]"
              << " loop_ms_avg=" << loop_stats.avg
              << " loop_ms_p95=" << loop_stats.p95
              << " loop_jitter_p95_vs_avg_pct=" << loop_jitter_pct
              << " instant_fps_avg=" << fps_stats.avg
              << " instant_fps_p95=" << fps_stats.p95
              << " fps_jitter_p95_vs_avg_pct=" << fps_jitter_pct
              << std::endl;
}

void PerformanceMonitor::ClearWindow() {
    get_image_ms_.clear();
    palm_total_ms_.clear();
    palm_preprocess_ms_.clear();
    palm_preprocess_transform_ms_.clear();
    palm_preprocess_manual_load_ms_.clear();
    palm_input_load_ms_.clear();
    palm_inference_ms_.clear();
    palm_getoutput_ms_.clear();
    palm_output_meta_ms_.clear();
    palm_decode_ms_.clear();
    palm_verbose_log_ms_.clear();
    palm_accounted_ms_.clear();
    hand_total_ms_.clear();
    draw_ms_.clear();
    loop_ms_.clear();
    process_ms_.clear();
    instant_fps_.clear();
}

}  // namespace slr_demo
