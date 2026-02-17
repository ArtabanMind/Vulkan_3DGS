// ============================================================
// 파일: src/main.cpp (수정본)
// 역할: GLFW 창 + Vulkan 초기화 + 구조체 테스트
// ============================================================
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cstdio>

#include "common/GaussianTypes.hpp"  // 새로 추가
#include "engine/VkEngine.hpp"
#include "engine/VkBuffer.hpp"

void checkGaussian() {
    // ----- 구조체 테스트 (Phase 0 검증) -----
    // 컴파일 + 실행 성공 시: "GaussianParam size: 16" 출력되어야 함
    // ----- 구조체 테스트 (Phase 0 검증) -----
    gs::GaussianParam g = gs::makeDefaultGaussian(
        glm::vec3(0.0f, 0.0f, 5.0f),   // position
        glm::vec3(1.0f, 0.0f, 0.0f)    // color: 빨강
    );
    
    printf("=== Phase 0: GaussianParam ===\n");
    printf("Size: %zu bytes (expect 64)\n", sizeof(g));
    printf("Position: (%.1f, %.1f, %.1f)\n", g.position.x, g.position.y, g.position.z);
    printf("Scale:    (%.1f, %.1f, %.1f)\n", g.scale.x, g.scale.y, g.scale.z);
    printf("Rotation: (%.1f, %.1f, %.1f, %.1f)\n", g.rotation.w, g.rotation.x, g.rotation.y, g.rotation.z);
    printf("Color:    (%.1f, %.1f, %.1f)\n", g.color.r, g.color.g, g.color.b);
    printf("Opacity:  %.1f\n", g.opacity);
}


int main() {
    // ----- GLFW 초기화 -----
    if (!glfwInit()) {
        printf("GLFW init failed\n");
        return -1;
    }

    // ----- Vulkan 지원 확인 -----
    if (!glfwVulkanSupported()) {
        printf("Vulkan not supported\n");
        glfwTerminate();
        return -1;
    }

    // ----- 윈도우 생성 (Vulkan용, OpenGL 컨텍스트 없음) -----
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Gaussian Splat", nullptr, nullptr);

    // ----- Vulkan extensions 출력 -----
    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    printf("Vulkan extensions: %u\n", extCount);


    gs::VkEngine engine;
    engine.init(window);

    // --- SSBO 테스트 ---
    printf("\n=== SSBO Test ===\n");

    // 테스트용 가우시안 생성
    gs::GaussianParam g = gs::makeDefaultGaussian(
        glm::vec3(0.0f, 0.0f, 5.0f),
        glm::vec3(1.0f, 0.0f, 0.0f)
    );

    // 가우시안 1개 담을 버퍼 생성 (HOST_VISIBLE로 간단하게)
    gs::BufferBundle ssbo = gs::createBuffer(
        engine.device(),
        engine.physicalDevice(),
        sizeof(gs::GaussianParam),                          // 64 bytes
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,                 // SSBO로 사용
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |               // CPU 접근 가능
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT                // 자동 동기화
    );
    printf("SSBO created: %zu bytes\n", (size_t)ssbo.size);

    // 데이터 업로드
    gs::uploadToBuffer(engine.device(), ssbo, &g, sizeof(g));
    printf("Data uploaded\n");

    // 데이터 다시 읽기 (검증)
    gs::GaussianParam readback{};
    gs::downloadFromBuffer(engine.device(), ssbo, &readback, sizeof(readback));
    printf("Readback position: (%.1f, %.1f, %.1f)\n", 
        readback.position.x, readback.position.y, readback.position.z);

    // 정리
    gs::destroyBuffer(engine.device(), ssbo);
    printf("SSBO destroyed\n");



    engine.cleanup();



    // ----- 메인 루프 -----
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
    }

    // ----- 정리 -----
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}