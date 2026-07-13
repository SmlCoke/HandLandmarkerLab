#pragma once

#include <array>
#include <string>

#include "osd-device.hpp"

class VISUALIZER {
public:
    void Initialize(std::array<int, 2>& in_img_shape, const std::string& bitmap_lut_path = "");
    void DrawPalmDetections(const PalmResult& result);
    void DrawDetections(const PalmResult& palm_result, const HandResult& hand_result);
    void Clear();
    void Release();

    void SetPointSize(int size) { point_size_ = size; }
    void SetPointColor(int color_idx) { point_color_ = color_idx; }
    void SetBoxBorder(int border) { box_border_ = border; }
    void SetBoxColor(int color_idx) { box_color_ = color_idx; }

private:
    bool IsInBounds(int x, int y) const;
    void DrawPalmBox(const std::array<float, 4>& box);
    void DrawHandSkeleton(const HandResult& result, int* draw_count);

    sst::device::osd::OsdDevice osd_device;
    int m_width = 0;
    int m_height = 0;
    int point_size_ = 5;
    int point_color_ = 2;
    int box_border_ = 6;
    int box_color_ = 2;
    int hand_line_thickness_ = 3;
    int hand_line_color_ = 3;
};
