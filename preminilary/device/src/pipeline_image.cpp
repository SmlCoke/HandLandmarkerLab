#include "../include/common.hpp"

#include <cstdio>

void IMAGEPROCESSOR::Initialize(std::array<int, 2>* in_img_shape) {
    img_shape = *in_img_shape;
    format_online = SSNE_Y_8;

    const uint16_t width = static_cast<uint16_t>(img_shape[0]);
    const uint16_t height = static_cast<uint16_t>(img_shape[1]);

    OnlineSetCrop(kPipeline0, 0, width, 0, height);
    OnlineSetOutputImage(kPipeline0, format_online, width, height);

    const int ret = OpenOnlinePipeline(kPipeline0);
    if (ret != 0) {
        printf("[IMAGEPROCESSOR] Failed to open full-frame online pipeline, ret=%d\n", ret);
        return;
    }

    printf("[IMAGEPROCESSOR] Opened full-frame gray pipeline: %ux%u, format=%u\n",
           width,
           height,
           format_online);
}

void IMAGEPROCESSOR::GetImage(ssne_tensor_t* img_sensor) {
    const int ret = GetImageData(img_sensor, kPipeline0, kSensor0, 0);
    if (ret != 0) {
        printf("[IMAGEPROCESSOR] Failed to get image from kPipeline0, ret=%d\n", ret);
    }
}

void IMAGEPROCESSOR::Release() {
    CloseOnlinePipeline(kPipeline0);
    printf("[IMAGEPROCESSOR] Online pipeline closed.\n");
}
