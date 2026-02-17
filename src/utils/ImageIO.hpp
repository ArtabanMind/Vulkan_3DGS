// ============================================================
// File: src/utils/ImageIO.hpp
// Role: PPM 이미지 저장 (디버깅용)
// ============================================================
#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <fstream>
#include <cstdio>
#include <algorithm>

namespace gs {

// ------------------------------------------------------------
// savePPM: float RGBA 버퍼 → PPM 파일
// ------------------------------------------------------------
// PPM = 가장 단순한 이미지 포맷 (헤더 + RGB bytes)
// 외부 라이브러리 불필요, 대부분 뷰어에서 열림
//
// pixels: vec4 배열 (RGBA, 각 채널 0~1)
// 예시: 64×64 이미지 → pixels.size() = 4096
// ------------------------------------------------------------
inline bool savePPM(
    const std::string& filename,
    const std::vector<glm::vec4>& pixels,
    uint32_t width,
    uint32_t height
) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        printf("[Error] Cannot open %s\n", filename.c_str());
        return false;
    }
    
    // PPM 헤더: P6 = binary RGB, width height, max value
    file << "P6\n" << width << " " << height << "\n255\n";
    
    // 픽셀 데이터 (RGB, top-to-bottom)
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            const glm::vec4& p = pixels[y * width + x];
            
            // float [0,1] → uint8 [0,255]
            uint8_t r = static_cast<uint8_t>(std::clamp(p.r, 0.0f, 1.0f) * 255.0f);
            uint8_t g = static_cast<uint8_t>(std::clamp(p.g, 0.0f, 1.0f) * 255.0f);
            uint8_t b = static_cast<uint8_t>(std::clamp(p.b, 0.0f, 1.0f) * 255.0f);
            
            file.write(reinterpret_cast<char*>(&r), 1);
            file.write(reinterpret_cast<char*>(&g), 1);
            file.write(reinterpret_cast<char*>(&b), 1);
        }
    }
    
    file.close();
    printf("[OK] Saved %s (%ux%u)\n", filename.c_str(), width, height);
    return true;
}

} // namespace gs