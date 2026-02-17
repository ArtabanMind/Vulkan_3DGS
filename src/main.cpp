// ============================================================
// File: src/main.cpp (Phase 2: Loss + Backward + 학습 루프)
// ============================================================
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cstdio>
#include <vector>
#include <cmath>

#include "common/GaussianTypes.hpp"
#include "engine/VkEngine.hpp"
#include "engine/VkBuffer.hpp"
#include "engine/VkCompute.hpp"
#include "utils/ImageIO.hpp"

// ============================================================
// Push Constants 구조체들
// ============================================================
struct RenderPC {
    uint32_t width;
    uint32_t height;
    uint32_t gaussCount;
};

struct LossPC {
    uint32_t width;
    uint32_t height;
};

// ============================================================
// GaussianGrad: gradient 저장용 (shader와 동일 레이아웃)
// ============================================================
struct GaussianGrad {
    glm::vec3 dPosition;
    float     dOpacity;
    glm::vec3 dScale;
    float     _pad0;
    glm::vec4 dRotation;
    glm::vec3 dColor;
    float     _pad1;
};
static_assert(sizeof(GaussianGrad) == 64, "GaussianGrad must be 64 bytes");

int main() {
    // ============================================================
    // GLFW 초기화
    // ============================================================
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Gaussian Splat", nullptr, nullptr);

    // ============================================================
    // Vulkan Engine
    // ============================================================
    gs::VkEngine engine;
    engine.init(window);

    // ============================================================
    // 이미지 설정
    // ============================================================
    const uint32_t IMG_W = 64;
    const uint32_t IMG_H = 64;
    const uint32_t pixelCount = IMG_W * IMG_H;
    const VkDeviceSize imageSize = pixelCount * sizeof(glm::vec4);
    const VkDeviceSize lossSize = pixelCount * sizeof(float);

    // ============================================================
    // Pipeline 생성
    // ============================================================
    printf("\n=== Create Pipelines ===\n");
    
    // gaussian.comp: binding 0=params, 1=image
    gs::ComputeContext renderPipeline = gs::createComputePipeline(
        engine.device(), "../src/shaders/gaussian.spv",
        2, sizeof(RenderPC)
    );
    
    // loss.comp: binding 0=rendered, 1=target, 2=pixelLoss
    gs::ComputeContext lossPipeline = gs::createComputePipeline(
        engine.device(), "../src/shaders/loss.spv",
        3, sizeof(LossPC)
    );

    // ============================================================
    // Target 이미지 생성 (CPU에서)
    // ============================================================
    // 목표: 중앙(32,32)에 초록색 가우시안
    // 학습: 빨간색 → 초록색, 위치 조정
    // ============================================================
    printf("\n=== Create Target Image ===\n");
    std::vector<glm::vec4> targetPixels(pixelCount);
    
    glm::vec2 targetCenter(32.0f, 32.0f);
    glm::vec3 targetColor(0.0f, 1.0f, 0.0f);  // 초록
    float targetSigma = 10.0f;
    
    for (uint32_t y = 0; y < IMG_H; y++) {
        for (uint32_t x = 0; x < IMG_W; x++) {
            glm::vec2 pos(x + 0.5f, y + 0.5f);
            glm::vec2 diff = pos - targetCenter;
            float r2 = glm::dot(diff, diff);
            float gaussian = std::exp(-0.5f * r2 / (targetSigma * targetSigma));
            
            targetPixels[y * IMG_W + x] = glm::vec4(targetColor * gaussian, 1.0f);
        }
    }
    gs::savePPM("target.ppm", targetPixels, IMG_W, IMG_H);

    // ============================================================
    // 버퍼 생성
    // ============================================================
    printf("\n=== Create Buffers ===\n");
    
    // params: 학습할 가우시안 (초기값: 빨강, 약간 벗어난 위치)
    gs::GaussianParam g = gs::makeDefaultGaussian(
        glm::vec3(28.0f, 28.0f, 0.0f),  // 중앙에서 벗어남
        glm::vec3(1.0f, 0.0f, 0.0f)     // 빨강 (목표는 초록)
    );
    g.scale = glm::vec3(10.0f);
    g.opacity = 1.0f;
    
    gs::BufferBundle paramsBuf = gs::createBuffer(
        engine.device(), engine.physicalDevice(),
        sizeof(gs::GaussianParam),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    
    // grads: gradient 저장
    gs::BufferBundle gradsBuf = gs::createBuffer(
        engine.device(), engine.physicalDevice(),
        sizeof(GaussianGrad),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    
    // rendered: forward 결과
    gs::BufferBundle renderedBuf = gs::createBuffer(
        engine.device(), engine.physicalDevice(),
        imageSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    
    // target: 목표 이미지
    gs::BufferBundle targetBuf = gs::createBuffer(
        engine.device(), engine.physicalDevice(),
        imageSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    gs::uploadToBuffer(engine.device(), targetBuf, targetPixels.data(), imageSize);
    
    // pixelLoss: 픽셀별 loss
    gs::BufferBundle lossBuf = gs::createBuffer(
        engine.device(), engine.physicalDevice(),
        lossSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // ============================================================
    // Descriptor 바인딩
    // ============================================================
    // renderPipeline: 0=params, 1=rendered
    gs::bindSSBO(engine.device(), renderPipeline, paramsBuf.buffer, paramsBuf.size, 0);
    gs::bindSSBO(engine.device(), renderPipeline, renderedBuf.buffer, renderedBuf.size, 1);
    
    // lossPipeline: 0=rendered, 1=target, 2=pixelLoss
    gs::bindSSBO(engine.device(), lossPipeline, renderedBuf.buffer, renderedBuf.size, 0);
    gs::bindSSBO(engine.device(), lossPipeline, targetBuf.buffer, targetBuf.size, 1);
    gs::bindSSBO(engine.device(), lossPipeline, lossBuf.buffer, lossBuf.size, 2);

    // ============================================================
    // 학습 루프
    // ============================================================
    printf("\n=== Training Loop ===\n");
    const int MAX_ITER = 100;
    const float lr = 0.5f;  // learning rate
    
    VkCommandBuffer cmd = engine.commandBuffer();
    
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // ---------- 파라미터 업로드 ----------
        gs::uploadToBuffer(engine.device(), paramsBuf, &g, sizeof(g));
        
        // ---------- gradient 초기화 ----------
        GaussianGrad grad{};
        gs::uploadToBuffer(engine.device(), gradsBuf, &grad, sizeof(grad));
        
        // ---------- Command Buffer 기록 ----------
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);
        
        // ----- Forward (gaussian.comp) -----
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, renderPipeline.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            renderPipeline.pipelineLayout, 0, 1, &renderPipeline.descriptorSet, 0, nullptr);
        RenderPC renderPC{ IMG_W, IMG_H, 1 };
        vkCmdPushConstants(cmd, renderPipeline.pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(renderPC), &renderPC);
        vkCmdDispatch(cmd, IMG_W / 8, IMG_H / 8, 1);
        
        // ----- 메모리 배리어 (Forward 완료 대기) -----
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
        
        // ----- Loss (loss.comp) -----
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lossPipeline.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            lossPipeline.pipelineLayout, 0, 1, &lossPipeline.descriptorSet, 0, nullptr);
        LossPC lossPC{ IMG_W, IMG_H };
        vkCmdPushConstants(cmd, lossPipeline.pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(lossPC), &lossPC);
        vkCmdDispatch(cmd, IMG_W / 8, IMG_H / 8, 1);
        
        vkEndCommandBuffer(cmd);
        
        // ---------- Submit + Wait ----------
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        vkQueueSubmit(engine.computeQueue(), 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(engine.computeQueue());
        
        // ---------- Loss 합산 (CPU) ----------
        std::vector<float> pixelLoss(pixelCount);
        gs::downloadFromBuffer(engine.device(), lossBuf, pixelLoss.data(), lossSize);
        
        float totalLoss = 0.0f;
        for (float l : pixelLoss) totalLoss += l;
        
        // ---------- Gradient 계산 (CPU, 간단 버전) ----------
        std::vector<glm::vec4> rendered(pixelCount);
        gs::downloadFromBuffer(engine.device(), renderedBuf, rendered.data(), imageSize);
        
        glm::vec3 dColor(0.0f);
        glm::vec3 dPosition(0.0f);
        
        for (uint32_t y = 0; y < IMG_H; y++) {
            for (uint32_t x = 0; x < IMG_W; x++) {
                uint32_t idx = y * IMG_W + x;
                
                // dL/dRendered
                glm::vec3 dL_dR = glm::vec3(rendered[idx]) - glm::vec3(targetPixels[idx]);
                
                // 가우시안 weight 재계산
                glm::vec2 pixelPos(x + 0.5f, y + 0.5f);
                glm::vec2 center(g.position.x, g.position.y);
                glm::vec2 diff = pixelPos - center;
                float r2 = glm::dot(diff, diff);
                float sigma2 = g.scale.x * g.scale.x;
                float gaussian = std::exp(-0.5f * r2 / sigma2);
                float alpha = gaussian * g.opacity;
                
                // dL/dColor = dL/dR * alpha
                dColor += dL_dR * alpha;
                
                // dL/dPosition = dL/dR * color * opacity * dGaussian/dPosition
                // dGaussian/dPos = gaussian * (diff / sigma²)
                glm::vec2 dGauss_dCenter = gaussian * diff / sigma2;  // +diff로 복원
                float dL_dGaussian = glm::dot(dL_dR, g.color) * g.opacity;
                
                dPosition.x += dL_dGaussian * dGauss_dCenter.x;
                dPosition.y += dL_dGaussian * dGauss_dCenter.y;
            }
        }
        
        // ---------- 파라미터 업데이트 ----------
        float colorLR = 0.5f;
        float posLR = 50.0f;
        g.color -= colorLR * dColor / float(pixelCount);
        g.position.x -= posLR * dPosition.x / float(pixelCount);
        g.position.y -= posLR * dPosition.y / float(pixelCount);
        
        // 색상 클램프
        g.color = glm::clamp(g.color, glm::vec3(0.0f), glm::vec3(1.0f));
        
        // ---------- 로그 ----------
        if (iter % 10 == 0 || iter == MAX_ITER - 1) {
            printf("Iter %3d | Loss: %.2f | Color: (%.2f, %.2f, %.2f) | Pos: (%.2f, %.2f)\n",
                iter, totalLoss,
                g.color.r, g.color.g, g.color.b,
                g.position.x, g.position.y);
        }
        
        // ---------- Command buffer 리셋 ----------
        vkResetCommandBuffer(cmd, 0);
    }

    // ============================================================
    // 최종 결과 저장
    // ============================================================
    printf("\n=== Save Final Result ===\n");
    std::vector<glm::vec4> finalImage(pixelCount);
    gs::downloadFromBuffer(engine.device(), renderedBuf, finalImage.data(), imageSize);
    gs::savePPM("final.ppm", finalImage, IMG_W, IMG_H);

    // ============================================================
    // Cleanup
    // ============================================================
    gs::destroyBuffer(engine.device(), paramsBuf);
    gs::destroyBuffer(engine.device(), gradsBuf);
    gs::destroyBuffer(engine.device(), renderedBuf);
    gs::destroyBuffer(engine.device(), targetBuf);
    gs::destroyBuffer(engine.device(), lossBuf);
    gs::destroyComputePipeline(engine.device(), renderPipeline);
    gs::destroyComputePipeline(engine.device(), lossPipeline);
    engine.cleanup();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
