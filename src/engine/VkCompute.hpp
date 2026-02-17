// ============================================================
// File: src/engine/VkCompute.hpp
// Role: Compute shader 로드 + 파이프라인 생성 + 실행
// Phase: 1-1 (Forward 테스트용, Backward 확장 고려)
// ============================================================
#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdio>

namespace gs {

// ------------------------------------------------------------
// SPIR-V 파일 로드
// ------------------------------------------------------------
// .spv = 컴파일된 shader 바이너리
// 예시: "shaders/simple.spv" → vector<char>로 읽음
// ------------------------------------------------------------
inline std::vector<char> loadSPV(const std::string& filename) {
    // ate: 파일 끝에서 시작 (크기 확인용)
    // binary: 바이너리 모드 (텍스트 변환 방지)
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader: " + filename);
    }
    
    size_t fileSize = (size_t)file.tellg();  // 현재 위치 = 파일 크기
    std::vector<char> buffer(fileSize);
    
    file.seekg(0);  // 처음으로 이동
    file.read(buffer.data(), fileSize);
    file.close();
    
    return buffer;
}

// ------------------------------------------------------------
// ComputeContext: Compute pipeline 관련 핸들 묶음
// ------------------------------------------------------------
// 학습 시 확장 고려:
//   - descriptorSetLayout에 binding 추가 (grads, targets)
//   - 여러 pipeline (forward, backward) 관리 가능
// ------------------------------------------------------------
struct ComputeContext {
    VkShaderModule       shaderModule       = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout     pipelineLayout     = VK_NULL_HANDLE;
    VkPipeline           pipeline           = VK_NULL_HANDLE;
    VkDescriptorPool     descriptorPool     = VK_NULL_HANDLE;
    VkDescriptorSet      descriptorSet      = VK_NULL_HANDLE;  // pool에서 할당, 개별 해제 불필요
};

// ------------------------------------------------------------
// createComputePipeline: 전체 파이프라인 생성
// ------------------------------------------------------------
// 순서: shader module → descriptor layout → pipeline layout → pipeline
//
// 학습 고려: binding 개수를 파라미터로 받으면 확장 용이
//           현재는 binding 0 (params) 하나만
// ------------------------------------------------------------
inline ComputeContext createComputePipeline(
    VkDevice device,
    const std::string& shaderPath
) {
    ComputeContext ctx;
    
    // ========== Step 1: Shader Module ==========
    // SPIR-V 바이너리 → VkShaderModule
    // 비유: 기계어 코드를 GPU가 이해하는 형태로 등록
    auto code = loadSPV(shaderPath);
    
    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = code.size();
    moduleInfo.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    
    if (vkCreateShaderModule(device, &moduleInfo, nullptr, &ctx.shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }
    printf("  [1/5] Shader module loaded\n");
    
    // ========== Step 2: Descriptor Set Layout ==========
    // SSBO 바인딩 정의 (shader의 layout(binding=0)과 매칭)
    //
    // 학습 확장 시:
    //   bindings[0] = params  (read/write)
    //   bindings[1] = grads   (write only)
    //   bindings[2] = target  (read only)
    VkDescriptorSetLayoutBinding binding{};
    binding.binding         = 0;  // shader의 binding = 0
    binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;  // SSBO
    binding.descriptorCount = 1;  // 버퍼 1개
    binding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;  // compute에서 사용
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings    = &binding;
    
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &ctx.descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
    printf("  [2/5] Descriptor set layout created\n");
    
    // ========== Step 3: Pipeline Layout ==========
    // Pipeline이 사용할 descriptor set layout 명시
    // Push constants도 여기서 정의 (현재는 안 씀)
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts    = &ctx.descriptorSetLayout;
    
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &ctx.pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }
    printf("  [3/5] Pipeline layout created\n");
    
    // ========== Step 4: Compute Pipeline ==========
    // Shader + Layout → 실행 가능한 파이프라인
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = ctx.pipelineLayout;
    
    // Shader stage 설정
    pipelineInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = ctx.shaderModule;
    pipelineInfo.stage.pName  = "main";  // shader entry point
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &ctx.pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline");
    }
    printf("  [4/5] Compute pipeline created\n");
    
    // ========== Step 5: Descriptor Pool ==========
    // Descriptor set 할당을 위한 메모리 풀
    // maxSets: 최대 할당 가능한 set 수 (학습 시 늘릴 수 있음)
    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 4;  // 여유있게 (forward + backward 대비)
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;
    poolInfo.maxSets       = 4;
    
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &ctx.descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }
    printf("  [5/5] Descriptor pool created\n");
    
    // ========== Step 6: Allocate Descriptor Set ==========
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = ctx.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &ctx.descriptorSetLayout;
    
    if (vkAllocateDescriptorSets(device, &allocInfo, &ctx.descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }
    printf("  [+] Descriptor set allocated\n");
    
    return ctx;
}

// ------------------------------------------------------------
// bindSSBO: Descriptor set에 SSBO 연결
// ------------------------------------------------------------
// 버퍼가 바뀔 때마다 호출 (또는 최초 1회)
// 학습 시: binding 번호를 파라미터로 받아 grads, targets도 바인딩
// ------------------------------------------------------------
inline void bindSSBO(
    VkDevice device,
    ComputeContext& ctx,
    VkBuffer buffer,
    VkDeviceSize size,
    uint32_t binding = 0
) {
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = 0;
    bufferInfo.range  = size;
    
    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = ctx.descriptorSet;
    write.dstBinding      = binding;
    write.dstArrayElement = 0;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo     = &bufferInfo;
    
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

// ------------------------------------------------------------
// destroyComputePipeline: 정리
// ------------------------------------------------------------
inline void destroyComputePipeline(VkDevice device, ComputeContext& ctx) {
    vkDestroyPipeline(device, ctx.pipeline, nullptr);
    vkDestroyPipelineLayout(device, ctx.pipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, ctx.descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, ctx.descriptorSetLayout, nullptr);
    vkDestroyShaderModule(device, ctx.shaderModule, nullptr);
}

} // namespace gs