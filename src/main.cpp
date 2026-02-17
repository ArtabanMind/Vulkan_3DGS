
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>
#include <cstdio>

#include "common/GaussianTypes.hpp"  // 새로 추가
#include "engine/VkEngine.hpp"
#include "engine/VkBuffer.hpp"
#include "engine/VkCompute.hpp"

// ============================================================
// 파일: src/main.cpp (수정본)
// 역할: GLFW 창 + Vulkan 초기화 + 구조체 테스트
// ============================================================

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
    // ===== GLFW + Window =====
    if (!glfwInit()) {
        printf("GLFW init failed\n");
        return -1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Gaussian Splat", nullptr, nullptr);

    // ===== Vulkan Engine =====
    gs::VkEngine engine;
    engine.init(window);

    // ===== Compute Pipeline =====
    printf("\n=== Compute Pipeline ===\n");
    gs::ComputeContext compute = gs::createComputePipeline(
        engine.device(),
        "../src/shaders/simple.spv"  // 컴파일된 shader
    );

    // ===== 테스트 데이터 준비 =====
    printf("\n=== Phase 1-1: Compute Test ===\n");
    
    // 초기값: position = (0, 0, 5), color = (1, 0, 0)
    gs::GaussianParam g = gs::makeDefaultGaussian(
        glm::vec3(0.0f, 0.0f, 5.0f),
        glm::vec3(1.0f, 0.0f, 0.0f)
    );
    printf("Before: position.x=%.1f, color.g=%.1f\n", g.position.x, g.color.g);

    // ===== SSBO 생성 + 업로드 =====
    gs::BufferBundle ssbo = gs::createBuffer(
        engine.device(),
        engine.physicalDevice(),
        sizeof(gs::GaussianParam),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    gs::uploadToBuffer(engine.device(), ssbo, &g, sizeof(g));

    // ===== SSBO를 Descriptor에 바인딩 =====
    gs::bindSSBO(engine.device(), compute, ssbo.buffer, ssbo.size, 0);

    // ===== Command Buffer 기록 =====
    VkCommandBuffer cmd = engine.commandBuffer();
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;  // 1회용
    
    vkBeginCommandBuffer(cmd, &beginInfo);
    
    // Pipeline 바인딩: "이 shader 사용할 거야"
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline);
    
    // Descriptor 바인딩: "이 SSBO 사용할 거야"
    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        compute.pipelineLayout,
        0,                      // firstSet
        1,                      // setCount
        &compute.descriptorSet,
        0, nullptr              // dynamic offsets (안 씀)
    );
    
    // Dispatch: "1개 workgroup 실행해"
    // 가우시안 1개 = workgroup 1개 (local_size=1이므로)
    // 나중에 N개면: vkCmdDispatch(cmd, N, 1, 1);
    vkCmdDispatch(cmd, 1, 1, 1);
    
    vkEndCommandBuffer(cmd);

    // ===== Submit + Wait =====
    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;
    
    vkQueueSubmit(engine.computeQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(engine.computeQueue());  // GPU 완료 대기

    // ===== Readback + 검증 =====
    gs::GaussianParam result{};
    gs::downloadFromBuffer(engine.device(), ssbo, &result, sizeof(result));
    
    printf("After:  position.x=%.1f, color.g=%.1f\n", result.position.x, result.color.g);
    
    // 검증: shader에서 position.x += 1.0, color.g += 0.5 했으므로
    bool pass = (result.position.x == 1.0f) && (result.color.g == 0.5f);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");

    // ===== Cleanup =====
    gs::destroyBuffer(engine.device(), ssbo);
    gs::destroyComputePipeline(engine.device(), compute);
    engine.cleanup();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}