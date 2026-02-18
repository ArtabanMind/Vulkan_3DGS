// ============================================================
// File: src/main.cpp (Phase 3: N개 가우시안 학습)
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
// Push Constants
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
// GaussianGrad
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

struct GaussianGradInt {
    glm::ivec3 dPosition;   int dOpacity;
    glm::ivec3 dScale;      int _pad0;
    glm::ivec4 dRotation;
    glm::ivec3 dColor;      int _pad1;
};

static_assert(sizeof(GaussianGradInt) == 64, "GaussianGradInt must be 64 bytes");

const float GRAD_SCALE = 1000000.0f;
// ============================================================
// CPU에서 가우시안 렌더링 (target 생성용)
// ============================================================
void renderGaussiansCPU(
    std::vector<glm::vec4>& pixels,
    const std::vector<gs::GaussianParam>& gaussians,
    uint32_t width, uint32_t height
) {
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            glm::vec2 pixelPos(x + 0.5f, y + 0.5f);
            
            glm::vec3 colorAccum(0.0f);
            float T = 1.0f;
            
            for (const auto& g : gaussians) {
                glm::vec2 center(g.position.x, g.position.y);
                glm::vec2 diff = pixelPos - center;
                float r2 = glm::dot(diff, diff);
                float sigma2 = g.scale.x * g.scale.x;
                float gaussian = std::exp(-0.5f * r2 / sigma2);
                float alpha = gaussian * g.opacity;
                
                colorAccum += g.color * alpha * T;
                T *= (1.0f - alpha);
                
                if (T < 0.001f) break;
            }
            
            pixels[y * width + x] = glm::vec4(colorAccum, 1.0f);
        }
    }
}

int main() {
    // ============================================================
    // 설정
    // ============================================================
    const uint32_t IMG_W = 64;
    const uint32_t IMG_H = 64;
    const uint32_t pixelCount = IMG_W * IMG_H;
    const uint32_t GAUSS_COUNT = 3;  // N개 가우시안
    
    const VkDeviceSize imageSize = pixelCount * sizeof(glm::vec4);
    const VkDeviceSize lossSize = pixelCount * sizeof(float);
    const VkDeviceSize paramsSize = GAUSS_COUNT * sizeof(gs::GaussianParam);
    const VkDeviceSize gradsSize = GAUSS_COUNT * sizeof(GaussianGrad);

    // ============================================================
    // GLFW + Vulkan
    // ============================================================
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Gaussian Splat", nullptr, nullptr);

    gs::VkEngine engine;
    engine.init(window);

    // ============================================================
    // Pipelines
    // ============================================================
    printf("\n=== Create Pipelines ===\n");
    gs::ComputeContext renderPipeline = gs::createComputePipeline(
        engine.device(), "../src/shaders/gaussian.spv", 2, sizeof(RenderPC));
    gs::ComputeContext lossPipeline = gs::createComputePipeline(
        engine.device(), "../src/shaders/loss.spv", 3, sizeof(LossPC));
    gs::ComputeContext backwardPipeline = gs::createComputePipeline(
        engine.device(), "../src/shaders/backward.spv", 4, sizeof(RenderPC));
    // ============================================================
    // Target 가우시안 (학습 목표)
    // ============================================================
    printf("\n=== Create Target ===\n");
    std::vector<gs::GaussianParam> targetGaussians = {
        gs::makeDefaultGaussian(glm::vec3(20.0f, 20.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)),  // 빨강
        gs::makeDefaultGaussian(glm::vec3(44.0f, 20.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),  // 초록
        gs::makeDefaultGaussian(glm::vec3(32.0f, 44.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),  // 파랑
    };
    for (auto& g : targetGaussians) {
        g.scale = glm::vec3(8.0f);
        g.opacity = 1.0f;
    }
    
    std::vector<glm::vec4> targetPixels(pixelCount);
    renderGaussiansCPU(targetPixels, targetGaussians, IMG_W, IMG_H);


    // ============================================================
    // 학습할 가우시안 (초기값: 위치/색상 랜덤하게 틀림)
    // ============================================================
    std::vector<gs::GaussianParam> gaussians = {
        gs::makeDefaultGaussian(glm::vec3(25.0f, 25.0f, 0.0f), glm::vec3(0.5f, 0.5f, 0.0f)),  // 노랑?
        gs::makeDefaultGaussian(glm::vec3(40.0f, 25.0f, 0.0f), glm::vec3(0.0f, 0.5f, 0.5f)),  // 청록?
        gs::makeDefaultGaussian(glm::vec3(30.0f, 40.0f, 0.0f), glm::vec3(0.5f, 0.0f, 0.5f)),  // 자주?
    };
    for (auto& g : gaussians) {
        g.scale = glm::vec3(8.0f);
        g.opacity = 1.0f;
    }

    // ============================================================
    // 버퍼 생성
    // ============================================================
    printf("\n=== Create Buffers ===\n");
    
    gs::BufferBundle paramsBuf = gs::createBuffer(
        engine.device(), engine.physicalDevice(),
        paramsSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    gs::BufferBundle gradsBuf = gs::createBuffer(
        engine.device(), engine.physicalDevice(),
        gradsSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    gs::BufferBundle renderedBuf = gs::createBuffer(
        engine.device(), engine.physicalDevice(),
        imageSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    gs::BufferBundle targetBuf = gs::createBuffer(
        engine.device(), engine.physicalDevice(),
        imageSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    gs::uploadToBuffer(engine.device(), targetBuf, targetPixels.data(), imageSize);
    
    gs::BufferBundle lossBuf = gs::createBuffer(
        engine.device(), engine.physicalDevice(),
        lossSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // ============================================================
    // Descriptor 바인딩
    // ============================================================
    gs::bindSSBO(engine.device(), renderPipeline, paramsBuf.buffer, paramsBuf.size, 0);
    gs::bindSSBO(engine.device(), renderPipeline, renderedBuf.buffer, renderedBuf.size, 1);
    
    gs::bindSSBO(engine.device(), lossPipeline, renderedBuf.buffer, renderedBuf.size, 0);
    gs::bindSSBO(engine.device(), lossPipeline, targetBuf.buffer, targetBuf.size, 1);
    gs::bindSSBO(engine.device(), lossPipeline, lossBuf.buffer, lossBuf.size, 2);

    gs::bindSSBO(engine.device(), backwardPipeline, paramsBuf.buffer, paramsBuf.size, 0); 
    gs::bindSSBO(engine.device(), backwardPipeline, gradsBuf.buffer, gradsBuf.size, 1);
    gs::bindSSBO(engine.device(), backwardPipeline, renderedBuf.buffer, renderedBuf.size,2);
    gs::bindSSBO(engine.device(), backwardPipeline, targetBuf.buffer, targetBuf.size, 3);
    // ============================================================
    // 학습 루프
    // ============================================================
    printf("\n=== Training Loop (N=%u) ===\n", GAUSS_COUNT);
    const int MAX_ITER = 200;
    const float colorLR = 0.3f;
    const float posLR = 30.0f;
    
    VkCommandBuffer cmd = engine.commandBuffer();
    
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // ---------- 파라미터 업로드 ----------
        gs::uploadToBuffer(engine.device(), paramsBuf, gaussians.data(), paramsSize);
        std::vector<GaussianGradInt> zeroGrads(GAUSS_COUNT, GaussianGradInt{});
        gs::uploadToBuffer(engine.device(), gradsBuf, zeroGrads.data(), gradsSize);

        // ---------- Command Buffer ----------
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);
        
        // Forward
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, renderPipeline.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            renderPipeline.pipelineLayout, 0, 1, &renderPipeline.descriptorSet, 0, nullptr);
        RenderPC renderPC{ IMG_W, IMG_H, GAUSS_COUNT };
        vkCmdPushConstants(cmd, renderPipeline.pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(renderPC), &renderPC);
        vkCmdDispatch(cmd, IMG_W / 8, IMG_H / 8, 1);
        
        // Barrier
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        
        // Loss
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lossPipeline.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            lossPipeline.pipelineLayout, 0, 1, &lossPipeline.descriptorSet, 0, nullptr);
        LossPC lossPC{ IMG_W, IMG_H };
        vkCmdPushConstants(cmd, lossPipeline.pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(lossPC), &lossPC);
        vkCmdDispatch(cmd, IMG_W / 8, IMG_H / 8, 1);
        
        // Backward
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, backwardPipeline.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            backwardPipeline.pipelineLayout, 0, 1, &backwardPipeline.descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, backwardPipeline.pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(renderPC), &renderPC);
        vkCmdDispatch(cmd, IMG_W / 8, IMG_H / 8, 1);
        
        vkEndCommandBuffer(cmd);

        // Processing ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        // Submit
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        vkQueueSubmit(engine.computeQueue(), 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(engine.computeQueue());
        // Processing ------------------------------------------------------
        
        // ---------- Loss 합산 ----------
        std::vector<float> pixelLoss(pixelCount);
        gs::downloadFromBuffer(engine.device(), lossBuf, pixelLoss.data(), lossSize);
        float totalLoss = 0.0f;
        for (float l : pixelLoss) totalLoss += l;
        
        // ---------- Gradient 계산 (CPU) ----------
        std::vector<glm::vec4> rendered(pixelCount);
        gs::downloadFromBuffer(engine.device(), renderedBuf, rendered.data(), imageSize);

        std::vector<GaussianGradInt> gradsInt(GAUSS_COUNT); 
        gs::downloadFromBuffer(engine.device(), gradsBuf, gradsInt.data(), gradsSize);

        // gradient 초기화
        std::vector<glm::vec3> dColors(GAUSS_COUNT, glm::vec3(0.0f));
        std::vector<glm::vec2> dPositions(GAUSS_COUNT, glm::vec2(0.0f));
        
        for (uint32_t i = 0; i < GAUSS_COUNT; i++) {
            glm::vec3 dColor = glm::vec3(gradsInt[i].dColor) / GRAD_SCALE / float(pixelCount);
            glm::vec2 dPos = glm::vec2(gradsInt[i].dPosition) / GRAD_SCALE / float(pixelCount);
            
            gaussians[i].color -= colorLR * dColor;
            gaussians[i].color = glm::clamp(gaussians[i].color, glm::vec3(0.0f), glm::vec3(1.0f));
            gaussians[i].position.x -= posLR * dPos.x;
            gaussians[i].position.y -= posLR * dPos.y;
        }

        // ---------- 로그 ----------
        if (iter % 20 == 0 || iter == MAX_ITER - 1) {
            printf("Iter %3d | Loss: %.2f\n", iter, totalLoss);
            for (uint32_t i = 0; i < GAUSS_COUNT; i++) {
                printf("  G%u: Color(%.2f,%.2f,%.2f) Pos(%.1f,%.1f)\n", i,
                    gaussians[i].color.r, gaussians[i].color.g, gaussians[i].color.b,
                    gaussians[i].position.x, gaussians[i].position.y);
            }
        }
        
        vkResetCommandBuffer(cmd, 0);
    }

    // ============================================================
    // 결과 저장
    // ============================================================
    printf("\n=== Save Results ===\n");
    std::vector<glm::vec4> finalImage(pixelCount);
    gs::downloadFromBuffer(engine.device(), renderedBuf, finalImage.data(), imageSize);
    gs::savePPM("../ppmOutput/final.ppm", finalImage, IMG_W, IMG_H);

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
    gs::destroyComputePipeline(engine.device(), backwardPipeline);
    engine.cleanup();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
