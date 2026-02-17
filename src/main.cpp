#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>
#include <cstdio>
#include <vector>

#include "common/GaussianTypes.hpp"
#include "engine/VkEngine.hpp"
#include "engine/VkBuffer.hpp"
#include "engine/VkCompute.hpp"
#include "utils/ImageIO.hpp"  // 새로 추가

// ============================================================
// File: src/main.cpp (Phase 1-1: 가우시안 블롭 렌더링)
// ============================================================

// Push constants 구조체 (shader와 동일해야 함)
struct PushConstants {
    uint32_t width;
    uint32_t height;
    uint32_t gaussCount;
};

int main() {
    // ===== GLFW =====
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Gaussian Splat", nullptr, nullptr);

    // ===== Vulkan Engine =====
    gs::VkEngine engine;
    engine.init(window);

    // ===== 이미지 설정 =====
    const uint32_t IMG_W = 64;
    const uint32_t IMG_H = 64;
    const uint32_t pixelCount = IMG_W * IMG_H;
    const VkDeviceSize imageSize = pixelCount * sizeof(glm::vec4);  // RGBA

    // ===== Compute Pipeline (2 bindings + push constants) =====
    printf("\n=== Gaussian Render Pipeline ===\n");
    gs::ComputeContext compute = gs::createComputePipeline(
        engine.device(),
        "../src/shaders/gaussian.spv",
        2,                      // binding 0: params, binding 1: image
        sizeof(PushConstants)   // 12 bytes
    );

    // ===== 가우시안 생성 (빨간 블롭, 중앙) =====
    gs::GaussianParam g = gs::makeDefaultGaussian(
        glm::vec3(32.0f, 32.0f, 0.0f),  // 중앙 (픽셀 좌표)
        glm::vec3(1.0f, 0.0f, 0.0f)     // 빨강
    );
    g.scale = glm::vec3(10.0f);  // 반지름 10 픽셀
    g.opacity = 1.0f;

    // ===== SSBO: params =====
    gs::BufferBundle paramsBuf = gs::createBuffer(
        engine.device(), engine.physicalDevice(),
        sizeof(gs::GaussianParam),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    gs::uploadToBuffer(engine.device(), paramsBuf, &g, sizeof(g));

    // ===== SSBO: image =====
    gs::BufferBundle imageBuf = gs::createBuffer(
        engine.device(), engine.physicalDevice(),
        imageSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // ===== Descriptor 바인딩 =====
    gs::bindSSBO(engine.device(), compute, paramsBuf.buffer, paramsBuf.size, 0);
    gs::bindSSBO(engine.device(), compute, imageBuf.buffer, imageBuf.size, 1);

    // ===== Command Buffer 기록 =====
    printf("\n=== Dispatch Compute ===\n");
    VkCommandBuffer cmd = engine.commandBuffer();
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, nullptr);
    
    // Push constants
    PushConstants pc{ IMG_W, IMG_H, 1 };
    vkCmdPushConstants(cmd, compute.pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    
    // Dispatch: 64/8 = 8 workgroups per axis
    vkCmdDispatch(cmd, IMG_W / 8, IMG_H / 8, 1);
    
    vkEndCommandBuffer(cmd);

    // ===== Submit + Wait =====
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(engine.computeQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(engine.computeQueue());
    printf("Compute finished\n");

    // ===== Readback + PPM 저장 =====
    std::vector<glm::vec4> pixels(pixelCount);
    gs::downloadFromBuffer(engine.device(), imageBuf, pixels.data(), imageSize);
    gs::savePPM("output.ppm", pixels, IMG_W, IMG_H);

    // ===== Cleanup =====
    gs::destroyBuffer(engine.device(), paramsBuf);
    gs::destroyBuffer(engine.device(), imageBuf);
    gs::destroyComputePipeline(engine.device(), compute);
    engine.cleanup();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
