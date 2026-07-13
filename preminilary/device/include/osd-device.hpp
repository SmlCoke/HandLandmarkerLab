#ifndef SST_OSD_DEVICE_HPP_
#define SST_OSD_DEVICE_HPP_

#include <array>
#include <string>
#include <vector>

#include "common.hpp"
#include "osd_lib_api.h"

#define BUFFER_TYPE_DMABUF 0x1
#define OSD_LAYER_SIZE 8

namespace sst {
namespace device {
namespace osd {

struct OsdQuadRangle {
    std::array<float, 4> box;
    int border;
    int layer_id;
    fdevice::QUADRANGLETYPE type;
    fdevice::ALPHATYPE alpha;
    int color;
};

class OsdDevice {
public:
    OsdDevice();
    ~OsdDevice();

    void Initialize(int width, int height, const char* bitmap_lut_path = nullptr);
    void Release();

    void DrawPoint(float x, float y, int size, int color_idx, int layer_id);
    void DrawLine(float x1, float y1, float x2, float y2, int thickness, int color_idx, int layer_id);
    void CleanLayer(int layer_id);
    void FlushLayer(int layer_id);

    void Draw(std::vector<OsdQuadRangle>& quad_rangle);
    void Draw(std::vector<OsdQuadRangle>& quad_rangle, int layer_id);
    void Draw(std::vector<std::array<float, 4>>& boxes,
              int border,
              int layer_id,
              fdevice::QUADRANGLETYPE type,
              fdevice::ALPHATYPE alpha,
              int color);
    void DrawTexture(const char* bitmap_path,
                     const char* lut_path,
                     int layer_id,
                     int pos_x = 0,
                     int pos_y = 0,
                     fdevice::ALPHATYPE alpha = fdevice::TYPE_ALPHA100);

private:
    int LoadLutFile(const char* filename);
    void GenQrangleBox(std::array<float, 4>& det, int border);
    int ClampX(int value) const;
    int ClampY(int value) const;
    bool IsGraphicLayer(int layer_id) const;

    handle_t m_osd_handle = INVALID_HANDLE;
    std::string m_osd_lut_path = "/app_demo/app_assets/colorLUT.sscl";
    uint8_t* m_pcolor_lut = nullptr;
    int m_file_size = 0;
    int m_height = 0;
    int m_width = 0;

    fdevice::DMA_BUFFER_ATTR_S m_layer_dma[OSD_LAYER_SIZE];
    fdevice::VERTEXS_S m_qrangle_out = {0};
    fdevice::VERTEXS_S m_qrangle_in = {0};
};

}  // namespace osd
}  // namespace device
}  // namespace sst

#endif  // SST_OSD_DEVICE_HPP_
