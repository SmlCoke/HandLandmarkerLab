#include "../include/utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

const int kHandConnectionCount = 21;
const int kHandConnections[kHandConnectionCount][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4},
    {0, 5}, {5, 6}, {6, 7}, {7, 8},
    {5, 9}, {9, 10}, {10, 11}, {11, 12},
    {9, 13}, {13, 14}, {14, 15}, {15, 16},
    {13, 17}, {17, 18}, {18, 19}, {19, 20},
    {0, 17},
};

}  // namespace

void VISUALIZER::Initialize(std::array<int, 2>& in_img_shape, const std::string& bitmap_lut_path) {
    m_width = in_img_shape[0];
    m_height = in_img_shape[1];

    std::string lut_path_full;
    const char* lut_path = nullptr;
    if (!bitmap_lut_path.empty()) {
        lut_path_full = "/app_demo/app_assets/" + bitmap_lut_path;
        lut_path = lut_path_full.c_str();
    }

    osd_device.Initialize(m_width, m_height, lut_path);
}

void VISUALIZER::DrawPalmDetections(const PalmResult& result) {
    HandResult empty_hand_result;
    DrawDetections(result, empty_hand_result);
}

void VISUALIZER::DrawDetections(const PalmResult& palm_result, const HandResult& hand_result) {
    Clear();
    if (!palm_result.valid && !hand_result.valid) {
        return;
    }

    if (palm_result.valid) {
        for (size_t i = 0; i < palm_result.detections.size(); i++) {
            DrawPalmBox(palm_result.detections[i].pixel_box);
        }
    }

    int draw_count = 0;
    if (palm_result.valid) {
        for (size_t det_idx = 0; det_idx < palm_result.detections.size(); det_idx++) {
            const PalmDetection& detection = palm_result.detections[det_idx];
            for (int kp = 0; kp < kPalmNumKeypoints; kp++) {
                const PalmKeypoint& point = detection.keypoints[kp];
                if (!IsInBounds(point.pixel_x, point.pixel_y)) {
                    continue;
                }

                const int layer = 1 + ((static_cast<int>(det_idx) * kPalmNumKeypoints + kp) %
                                       (OSD_LAYER_SIZE - 2));
                osd_device.DrawPoint(static_cast<float>(point.pixel_x),
                                     static_cast<float>(point.pixel_y),
                                     point_size_,
                                     point_color_,
                                     layer);
            }
        }
    }

    DrawHandSkeleton(hand_result, &draw_count);

    for (int layer = 0; layer < OSD_LAYER_SIZE - 1; layer++) {
        osd_device.FlushLayer(layer);
    }
}

void VISUALIZER::Clear() {
    for (int layer = 0; layer < OSD_LAYER_SIZE - 1; layer++) {
        osd_device.CleanLayer(layer);
        osd_device.FlushLayer(layer);
    }
}

void VISUALIZER::Release() {
    osd_device.Release();
}

bool VISUALIZER::IsInBounds(int x, int y) const {
    return x >= 0 && x < m_width && y >= 0 && y < m_height;
}

void VISUALIZER::DrawPalmBox(const std::array<float, 4>& box) {
    int x1 = static_cast<int>(std::round(std::min(box[0], box[2])));
    int y1 = static_cast<int>(std::round(std::min(box[1], box[3])));
    int x2 = static_cast<int>(std::round(std::max(box[0], box[2])));
    int y2 = static_cast<int>(std::round(std::max(box[1], box[3])));

    x1 = std::max(0, std::min(x1, m_width - 1));
    y1 = std::max(0, std::min(y1, m_height - 1));
    x2 = std::max(0, std::min(x2, m_width - 1));
    y2 = std::max(0, std::min(y2, m_height - 1));

    if (x2 <= x1 || y2 <= y1) {
        return;
    }

    const int layer = 0;
    osd_device.DrawLine(static_cast<float>(x1),
                        static_cast<float>(y1),
                        static_cast<float>(x2),
                        static_cast<float>(y1),
                        box_border_,
                        box_color_,
                        layer);
    osd_device.DrawLine(static_cast<float>(x2),
                        static_cast<float>(y1),
                        static_cast<float>(x2),
                        static_cast<float>(y2),
                        box_border_,
                        box_color_,
                        layer);
    osd_device.DrawLine(static_cast<float>(x2),
                        static_cast<float>(y2),
                        static_cast<float>(x1),
                        static_cast<float>(y2),
                        box_border_,
                        box_color_,
                        layer);
    osd_device.DrawLine(static_cast<float>(x1),
                        static_cast<float>(y2),
                        static_cast<float>(x1),
                        static_cast<float>(y1),
                        box_border_,
                        box_color_,
                        layer);
}

void VISUALIZER::DrawHandSkeleton(const HandResult& result, int* draw_count) {
    if (!result.valid || draw_count == nullptr) {
        return;
    }

    const int graphic_layers = OSD_LAYER_SIZE - 1;
    for (size_t det_idx = 0; det_idx < result.detections.size(); det_idx++) {
        const HandDetection& hand = result.detections[det_idx];
        if (!hand.valid) {
            continue;
        }

        for (int conn_idx = 0; conn_idx < kHandConnectionCount; conn_idx++) {
            const int a = kHandConnections[conn_idx][0];
            const int b = kHandConnections[conn_idx][1];
            if (a < 0 || a >= kHandNumLandmarks || b < 0 || b >= kHandNumLandmarks) {
                continue;
            }

            const HandLandmark& p1 = hand.landmarks[a];
            const HandLandmark& p2 = hand.landmarks[b];
            if (!IsInBounds(p1.pixel_x, p1.pixel_y) || !IsInBounds(p2.pixel_x, p2.pixel_y)) {
                continue;
            }

            const int layer = *draw_count % graphic_layers;
            osd_device.DrawLine(static_cast<float>(p1.pixel_x),
                                static_cast<float>(p1.pixel_y),
                                static_cast<float>(p2.pixel_x),
                                static_cast<float>(p2.pixel_y),
                                hand_line_thickness_,
                                hand_line_color_,
                                layer);
            *draw_count += 1;
        }
    }
}
