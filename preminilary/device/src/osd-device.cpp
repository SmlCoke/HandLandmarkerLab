#include "../include/osd-device.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

using namespace fdevice;

namespace sst {
namespace device {
namespace osd {

OsdDevice::OsdDevice() {}

OsdDevice::~OsdDevice() {}

void OsdDevice::Initialize(int width, int height, const char* bitmap_lut_path) {
    m_width = width;
    m_height = height;

    const char* lut_path = bitmap_lut_path;
    if (lut_path == nullptr || strlen(lut_path) == 0 || LoadLutFile(lut_path) != 0) {
        if (lut_path != nullptr && strlen(lut_path) > 0) {
            std::cerr << "[OsdDevice] Failed to load requested LUT, falling back to default LUT." << std::endl;
        }
        if (LoadLutFile(m_osd_lut_path.c_str()) != 0) {
            std::cerr << "[OsdDevice] Failed to load OSD LUT: " << m_osd_lut_path << std::endl;
        }
    }

    m_osd_handle = osd_open_device();
    osd_init_device(m_osd_handle, OSD_LAYER_SIZE, reinterpret_cast<char*>(m_pcolor_lut));

    const int graphic_dma_size = 2048;
    for (int layer_index = 0; layer_index < OSD_LAYER_SIZE - 1; layer_index++) {
        osd_alloc_buffer(m_osd_handle, m_layer_dma[layer_index].dma, graphic_dma_size);
        usleep(250000);
        osd_alloc_buffer(m_osd_handle, m_layer_dma[layer_index].dma_2, graphic_dma_size);

        const int dma_fd = osd_get_buffer_fd(m_osd_handle, m_layer_dma[layer_index].dma);
        LAYER_ATTR_S osd_layer = {};
        osd_layer.codeTYPE = SS_TYPE_QUADRANGLE;
        osd_layer.layer_data_QR.osd_buf.buf_type = BUFFER_TYPE_DMABUF;
        osd_layer.layer_data_QR.osd_buf.buf.fd_dmabuf = dma_fd;
        osd_layer.layerStart.layer_start_x = 0;
        osd_layer.layerStart.layer_start_y = 0;
        osd_layer.layerSize.layer_width = m_width;
        osd_layer.layerSize.layer_height = m_height;
        osd_layer.layer_rgn = {TYPE_GRAPHIC, {m_width, m_height}};

        osd_create_layer(m_osd_handle, static_cast<ssLAYER_HANDLE>(layer_index), &osd_layer);
        osd_set_layer_buffer(m_osd_handle, static_cast<ssLAYER_HANDLE>(layer_index), m_layer_dma[layer_index]);
    }

    const int image_layer = OSD_LAYER_SIZE - 1;
    osd_alloc_buffer(m_osd_handle, m_layer_dma[image_layer].dma, 0x20000);
    usleep(250000);
    osd_alloc_buffer(m_osd_handle, m_layer_dma[image_layer].dma_2, 0x20000);

    const int dma_fd = osd_get_buffer_fd(m_osd_handle, m_layer_dma[image_layer].dma);
    LAYER_ATTR_S osd_layer = {};
    osd_layer.codeTYPE = SS_TYPE_RLE;
    osd_layer.layer_data_RLE.osd_buf.buf_type = BUFFER_TYPE_DMABUF;
    osd_layer.layer_data_RLE.osd_buf.buf.fd_dmabuf = dma_fd;
    osd_layer.layerStart.layer_start_x = 0;
    osd_layer.layerStart.layer_start_y = 0;
    osd_layer.layerSize.layer_width = m_width;
    osd_layer.layerSize.layer_height = m_height;
    osd_layer.layer_rgn = {TYPE_IMAGE, {m_width, m_height}};
    osd_create_layer(m_osd_handle, static_cast<ssLAYER_HANDLE>(image_layer), &osd_layer);
    osd_set_layer_buffer(m_osd_handle, static_cast<ssLAYER_HANDLE>(image_layer), m_layer_dma[image_layer]);

    std::cout << "[OsdDevice] Initialized " << OSD_LAYER_SIZE
              << " layers: layers 0-" << (OSD_LAYER_SIZE - 2)
              << " are graphic, layer " << image_layer << " is image." << std::endl;
}

void OsdDevice::Release() {
    for (int i = 0; i < OSD_LAYER_SIZE; i++) {
        osd_destroy_layer(m_osd_handle, static_cast<ssLAYER_HANDLE>(i));
        if (m_layer_dma[i].dma != nullptr) {
            osd_delete_buffer(m_osd_handle, m_layer_dma[i].dma);
            m_layer_dma[i].dma = nullptr;
        }
        if (m_layer_dma[i].dma_2 != nullptr) {
            osd_delete_buffer(m_osd_handle, m_layer_dma[i].dma_2);
            m_layer_dma[i].dma_2 = nullptr;
        }
    }

    if (m_pcolor_lut != nullptr) {
        delete[] m_pcolor_lut;
        m_pcolor_lut = nullptr;
    }

    if (m_osd_handle != INVALID_HANDLE) {
        osd_close_device(m_osd_handle);
        m_osd_handle = INVALID_HANDLE;
    }
}

void OsdDevice::DrawPoint(float x, float y, int size, int color_idx, int layer_id) {
    if (!IsGraphicLayer(layer_id)) {
        return;
    }

    COVER_ATTR_S attr = {};
    attr.colorIdx = color_idx;
    attr.eSolid = TYPE_SOLID;
    attr.alpha = TYPE_ALPHA100;

    attr.vertex_out.points[0] = {ClampX(static_cast<int>(x - size)), ClampY(static_cast<int>(y - size))};
    attr.vertex_out.points[1] = {ClampX(static_cast<int>(x + size)), ClampY(static_cast<int>(y - size))};
    attr.vertex_out.points[2] = {ClampX(static_cast<int>(x + size)), ClampY(static_cast<int>(y + size))};
    attr.vertex_out.points[3] = {ClampX(static_cast<int>(x - size)), ClampY(static_cast<int>(y + size))};
    attr.vertex_in = attr.vertex_out;

    osd_add_quad_rangle_layer(m_osd_handle, static_cast<ssLAYER_HANDLE>(layer_id), &attr);
}

void OsdDevice::DrawLine(float x1, float y1, float x2, float y2, int thickness, int color_idx, int layer_id) {
    if (!IsGraphicLayer(layer_id)) {
        return;
    }

    const float dx = x2 - x1;
    const float dy = y2 - y1;
    const float length = std::sqrt(dx * dx + dy * dy);
    if (length < 1.0f) {
        return;
    }

    const float nx = -dy / length;
    const float ny = dx / length;
    const float half_t = static_cast<float>(thickness) / 2.0f;

    COVER_ATTR_S attr = {};
    attr.colorIdx = color_idx;
    attr.eSolid = TYPE_SOLID;
    attr.alpha = TYPE_ALPHA100;

    attr.vertex_out.points[0] = {ClampX(static_cast<int>(x1 + nx * half_t)), ClampY(static_cast<int>(y1 + ny * half_t))};
    attr.vertex_out.points[1] = {ClampX(static_cast<int>(x2 + nx * half_t)), ClampY(static_cast<int>(y2 + ny * half_t))};
    attr.vertex_out.points[2] = {ClampX(static_cast<int>(x2 - nx * half_t)), ClampY(static_cast<int>(y2 - ny * half_t))};
    attr.vertex_out.points[3] = {ClampX(static_cast<int>(x1 - nx * half_t)), ClampY(static_cast<int>(y1 - ny * half_t))};
    attr.vertex_in = attr.vertex_out;

    osd_add_quad_rangle_layer(m_osd_handle, static_cast<ssLAYER_HANDLE>(layer_id), &attr);
}

void OsdDevice::CleanLayer(int layer_id) {
    if (layer_id < 0 || layer_id >= OSD_LAYER_SIZE) {
        return;
    }
    osd_clean_layer(m_osd_handle, static_cast<ssLAYER_HANDLE>(layer_id));
}

void OsdDevice::FlushLayer(int layer_id) {
    if (!IsGraphicLayer(layer_id)) {
        return;
    }
    osd_flush_quad_rangle_layer(m_osd_handle, static_cast<ssLAYER_HANDLE>(layer_id));
}

int OsdDevice::LoadLutFile(const char* filename) {
    struct stat file_stat;
    if (filename == nullptr || stat(filename, &file_stat) != 0 || file_stat.st_size <= 0) {
        if (filename != nullptr) {
            std::cerr << "[OsdDevice] LUT not available: " << filename
                      << ", errno=" << errno << " (" << strerror(errno) << ")" << std::endl;
        }
        return -1;
    }

    std::ifstream file(filename, std::ios::binary | std::ios::in | std::ios::ate);
    if (!file) {
        std::cerr << "[OsdDevice] Cannot open LUT file: " << filename << std::endl;
        return -1;
    }

    if (m_pcolor_lut != nullptr) {
        delete[] m_pcolor_lut;
        m_pcolor_lut = nullptr;
    }

    m_file_size = static_cast<int>(file.tellg());
    m_pcolor_lut = new uint8_t[m_file_size];
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(m_pcolor_lut), m_file_size);
    const bool ok = file.good() || file.eof();
    file.close();

    if (!ok) {
        delete[] m_pcolor_lut;
        m_pcolor_lut = nullptr;
        m_file_size = 0;
        return -1;
    }

    std::cout << "[OsdDevice] Loaded LUT: " << filename << ", bytes=" << m_file_size << std::endl;
    return 0;
}

void OsdDevice::Draw(std::vector<OsdQuadRangle>& quad_rangle) {
    if (quad_rangle.empty()) {
        osd_clean_all_layer(m_osd_handle);
        return;
    }

    for (auto& q : quad_rangle) {
        GenQrangleBox(q.box, q.border);
        COVER_ATTR_S attr = {q.color, q.type, q.alpha, m_qrangle_out, m_qrangle_in};
        osd_add_quad_rangle(m_osd_handle, &attr);
    }
    osd_flush_quad_rangle(m_osd_handle);
}

void OsdDevice::Draw(std::vector<OsdQuadRangle>& quad_rangle, int layer_id) {
    if (!IsGraphicLayer(layer_id)) {
        return;
    }
    if (quad_rangle.empty()) {
        CleanLayer(layer_id);
        return;
    }

    for (auto& q : quad_rangle) {
        GenQrangleBox(q.box, q.border);
        COVER_ATTR_S attr = {q.color, q.type, q.alpha, m_qrangle_out, m_qrangle_in};
        osd_add_quad_rangle_layer(m_osd_handle, static_cast<ssLAYER_HANDLE>(layer_id), &attr);
    }
    osd_flush_quad_rangle_layer(m_osd_handle, static_cast<ssLAYER_HANDLE>(layer_id));
}

void OsdDevice::Draw(std::vector<std::array<float, 4>>& boxes,
                     int border,
                     int layer_id,
                     fdevice::QUADRANGLETYPE type,
                     fdevice::ALPHATYPE alpha,
                     int color) {
    if (!IsGraphicLayer(layer_id)) {
        return;
    }
    if (boxes.empty()) {
        CleanLayer(layer_id);
        return;
    }

    for (auto& box : boxes) {
        GenQrangleBox(box, border);
        COVER_ATTR_S attr = {color, type, alpha, m_qrangle_out, m_qrangle_in};
        osd_add_quad_rangle_layer(m_osd_handle, static_cast<ssLAYER_HANDLE>(layer_id), &attr);
    }
    osd_flush_quad_rangle_layer(m_osd_handle, static_cast<ssLAYER_HANDLE>(layer_id));
}

void OsdDevice::DrawTexture(const char* bitmap_path,
                            const char*,
                            int layer_id,
                            int pos_x,
                            int pos_y,
                            fdevice::ALPHATYPE alpha) {
    if (layer_id < 0 || layer_id >= OSD_LAYER_SIZE) {
        return;
    }

    fdevice::BITMAP_INFO_S bm_info = {};
    bm_info.pSSbmpFile = bitmap_path;
    bm_info.alpha = alpha;
    bm_info.position.x = pos_x;
    bm_info.position.y = pos_y;

    osd_add_texture_layer(m_osd_handle, static_cast<ssLAYER_HANDLE>(layer_id), &bm_info);
    osd_flush_texture_layer(m_osd_handle, static_cast<ssLAYER_HANDLE>(layer_id));
}

void OsdDevice::GenQrangleBox(std::array<float, 4>& det, int border) {
    std::array<int, 16> box;

    box[0] = ClampX(static_cast<int>(det[0] + border));
    box[1] = ClampY(static_cast<int>(det[1] + border));
    box[2] = ClampX(static_cast<int>(det[0] + border));
    box[3] = ClampY(static_cast<int>(det[3] - border));
    box[4] = ClampX(static_cast<int>(det[2] - border));
    box[5] = ClampY(static_cast<int>(det[3] - border));
    box[6] = ClampX(static_cast<int>(det[2] - border));
    box[7] = ClampY(static_cast<int>(det[1] + border));

    box[8] = ClampX(static_cast<int>(det[0] - border));
    box[9] = ClampY(static_cast<int>(det[1] - border));
    box[10] = ClampX(static_cast<int>(det[0] - border));
    box[11] = ClampY(static_cast<int>(det[3] + border));
    box[12] = ClampX(static_cast<int>(det[2] + border));
    box[13] = ClampY(static_cast<int>(det[3] + border));
    box[14] = ClampX(static_cast<int>(det[2] + border));
    box[15] = ClampY(static_cast<int>(det[1] - border));

    m_qrangle_in.points[0] = {box[0], box[1]};
    m_qrangle_in.points[1] = {box[2], box[3]};
    m_qrangle_in.points[2] = {box[4], box[5]};
    m_qrangle_in.points[3] = {box[6], box[7]};
    m_qrangle_out.points[0] = {box[8], box[9]};
    m_qrangle_out.points[1] = {box[10], box[11]};
    m_qrangle_out.points[2] = {box[12], box[13]};
    m_qrangle_out.points[3] = {box[14], box[15]};
}

int OsdDevice::ClampX(int value) const {
    return std::max(0, std::min(m_width - 1, value));
}

int OsdDevice::ClampY(int value) const {
    return std::max(0, std::min(m_height - 1, value));
}

bool OsdDevice::IsGraphicLayer(int layer_id) const {
    return layer_id >= 0 && layer_id < OSD_LAYER_SIZE - 1;
}

}  // namespace osd
}  // namespace device
}  // namespace sst
